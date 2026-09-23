#include <pybind11/pybind11.h>
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"

namespace py = pybind11;
using namespace mlir;

template <typename T> std::string printed(T value) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  value.print(stream);
  return text;
}

py::dict typeSnapshot(Type type) {
  py::dict result;
  result["assembly"] = printed(type);
  if (auto tensor = dyn_cast<RankedTensorType>(type)) {
    result["kind"] = "tensor";
    py::list shape;
    for (auto size : tensor.getShape()) shape.append(size);
    result["shape"] = shape;
    result["element"] = typeSnapshot(tensor.getElementType());
    result["encoding"] = tensor.getEncoding() ? printed(tensor.getEncoding()) : "";
  } else if (type.getAbstractType().getName() == "tt.ptr") {
    Type pointee;
    type.walkImmediateSubElements([](Attribute) {}, [&](Type child) { pointee = child; });
    if (!pointee) throw std::runtime_error("pointer type has no structural pointee");
    result["kind"] = "pointer";
    result["element"] = typeSnapshot(pointee);
    // Triton's pointer accessors are hidden symbols. Compare official type objects
    // to the default-address-space type; all other address spaces fail closed.
    Type expected = parseType("!tt.ptr<" + printed(pointee) + ">", type.getContext());
    result["address_space"] = type == expected ? 1 : -1;
  } else if (auto integer = dyn_cast<IntegerType>(type)) {
    result["kind"] = "int";
    result["bits"] = integer.getWidth();
    result["signedness"] = static_cast<int>(integer.getSignedness());
  } else if (auto floating = dyn_cast<FloatType>(type)) {
    result["kind"] = "float";
    result["bits"] = floating.getWidth();
  } else {
    result["kind"] = type.isIndex() ? "index" : "unsupported";
  }
  return result;
}

py::str integerValue(const llvm::APInt &value) {
  llvm::SmallString<80> text;
  value.toString(text, 10, true);
  return py::str(text.str().str());
}

py::object floatValue(const llvm::APFloat &value) {
  if (!value.isFinite() || llvm::APFloat::semanticsSizeInBits(value.getSemantics()) > 64)
    return py::none();
  llvm::APFloat converted(value);
  bool losesInfo;
  converted.convert(llvm::APFloat::IEEEdouble(), llvm::APFloat::rmNearestTiesToEven, &losesInfo);
  if (losesInfo) return py::none();
  return py::float_(converted.convertToDouble());
}

py::dict attrSnapshot(Attribute attr) {
  py::dict result;
  result["assembly"] = printed(attr);
  if (auto integer = dyn_cast<IntegerAttr>(attr)) {
    result["kind"] = "integer";
    result["value"] = integerValue(integer.getValue());
  } else if (auto floating = dyn_cast<FloatAttr>(attr)) {
    result["kind"] = "float";
    result["value"] = floatValue(floating.getValue());
  } else if (auto string = dyn_cast<StringAttr>(attr)) {
    result["kind"] = "string";
    result["value"] = string.getValue().str();
  } else if (auto dense = dyn_cast<DenseElementsAttr>(attr)) {
    result["kind"] = "dense";
    result["splat"] = dense.isSplat();
    py::list values;
    int64_t count = dense.isSplat() ? 1 : dense.getNumElements();
    if (count > 100000) throw std::runtime_error("RESOURCE_LIMIT: dense constant");
    if (dense.getElementType().isIntOrIndex()) {
      for (auto value : dense.getValues<llvm::APInt>()) {
        values.append(integerValue(value));
        if (--count == 0) break;
      }
    } else if (isa<FloatType>(dense.getElementType())) {
      for (auto value : dense.getValues<llvm::APFloat>()) {
        values.append(floatValue(value));
        if (--count == 0) break;
      }
    }
    result["value"] = values;
  } else if (auto flags = dyn_cast<arith::IntegerOverflowFlagsAttr>(attr)) {
    result["kind"] = "integer";
    result["value"] = std::to_string(static_cast<unsigned>(flags.getValue()));
  } else if (auto flags = dyn_cast<arith::FastMathFlagsAttr>(attr)) {
    result["kind"] = "integer";
    result["value"] = std::to_string(static_cast<unsigned>(flags.getValue()));
  } else if (auto array = dyn_cast<ArrayAttr>(attr)) {
    result["kind"] = "array";
    py::list values;
    for (auto value : array) values.append(attrSnapshot(value));
    result["value"] = values;
  } else if (auto dict = dyn_cast<DictionaryAttr>(attr)) {
    result["kind"] = "dictionary";
    py::dict values;
    for (auto named : dict) values[py::str(named.getName().str())] = attrSnapshot(named.getValue());
    result["value"] = values;
  } else {
    result["kind"] = "opaque";
  }
  return result;
}

struct Snapshot {
  llvm::DenseMap<Value, unsigned> values;
  llvm::DenseMap<Block *, unsigned> blocks;
  unsigned ordinal = 0;

  void index(Operation *op) {
    for (auto value : op->getResults()) values.try_emplace(value, values.size());
    for (auto &region : op->getRegions()) {
      for (auto &block : region) {
        blocks.try_emplace(&block, blocks.size());
        for (auto arg : block.getArguments()) values.try_emplace(arg, values.size());
        for (auto &child : block) index(&child);
      }
    }
  }

  py::dict value(Value value) {
    py::dict result;
    result["id"] = "v" + std::to_string(values.lookup(value));
    result["type"] = typeSnapshot(value.getType());
    return result;
  }

  py::dict operation(Operation *op) {
    py::dict result;
    result["ordinal"] = ordinal++;
    result["name"] = op->getName().getStringRef().str();
    result["block"] = op->getBlock() ? py::cast(blocks.lookup(op->getBlock())) : py::none();
    result["location"] = printed(op->getLoc());
    std::string assembly;
    llvm::raw_string_ostream stream(assembly);
    op->print(stream, OpPrintingFlags().printGenericOpForm().useLocalScope());
    result["assembly"] = assembly;
    py::list operands, results, regions;
    for (auto operand : op->getOperands()) operands.append(value(operand));
    for (auto output : op->getResults()) results.append(value(output));
    result["operands"] = operands;
    result["results"] = results;
    py::dict attrs;
    for (auto named : op->getAttrs())
      attrs[py::str(named.getName().str())] = attrSnapshot(named.getValue());
    result["attributes"] = attrs;
    for (auto &region : op->getRegions()) {
      py::list regionBlocks;
      for (auto &block : region) {
        py::dict blockResult;
        py::list args, operations;
        blockResult["id"] = blocks.lookup(&block);
        for (auto arg : block.getArguments()) args.append(value(arg));
        for (auto &child : block) operations.append(operation(&child));
        blockResult["arguments"] = args;
        blockResult["operations"] = operations;
        regionBlocks.append(blockResult);
      }
      regions.append(regionBlocks);
    }
    result["regions"] = regions;
    return result;
  }
};

PYBIND11_MODULE(_mlir_native, module) {
  module.attr("triton_version") = "3.7.1";
  module.def("snapshot", [](ModuleOp &module) {
    if (failed(verify(module))) throw std::runtime_error("TTIR_VERIFY_FAILED");
    Snapshot snapshot;
    snapshot.index(module.getOperation());
    return snapshot.operation(module.getOperation());
  });
}
