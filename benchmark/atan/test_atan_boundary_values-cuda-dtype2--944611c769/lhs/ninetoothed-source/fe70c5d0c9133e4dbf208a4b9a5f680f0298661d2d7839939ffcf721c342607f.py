import triton
import triton.language

@triton.jit
def application(ninetoothed_ninetoothed_tensor_0_pointer, ninetoothed_ninetoothed_tensor_0_size_0, ninetoothed_ninetoothed_tensor_0_stride_0, ninetoothed_ninetoothed_tensor_1_pointer, ninetoothed_ninetoothed_tensor_1_size_0, ninetoothed_ninetoothed_tensor_1_stride_0):
    ninetoothed_pid = triton.language.program_id(0)
    ninetoothed_tensor_0_index_0 = ninetoothed_pid
    ninetoothed_tensor_0_pointers = ninetoothed_ninetoothed_tensor_0_pointer
    ninetoothed_tensor_1_index_0 = ninetoothed_pid
    ninetoothed_tensor_1_pointers = ninetoothed_ninetoothed_tensor_1_pointer
    '\n    计算反正切函数 atan(x)\n\n    参数:\n    input: 输入张量，形状为 (C // block_size, block_size)\n    output: 输出张量，形状为 (C // block_size, block_size)\n    '
    dtype = ninetoothed_ninetoothed_tensor_1_pointer.type.element_ty
    for i in range(((ninetoothed_ninetoothed_tensor_0_size_0 - 255 - 1 + 256 - 1) // 256 + 1,)[0]):
        input_block = triton.language.cast(triton.language.load(ninetoothed_tensor_0_pointers + ((ninetoothed_tensor_0_index_0 * ((ninetoothed_ninetoothed_tensor_0_size_0 - 255 - 1 + 256 - 1) // 256 + 1) + i) * 256 + triton.language.arange(0, 256)[::,]) * ninetoothed_ninetoothed_tensor_0_stride_0, mask=True & (triton.language.arange(0, 256)[::,] < 256) & (triton.language.arange(0, 256)[::,] >= 0) & (i < (ninetoothed_ninetoothed_tensor_0_size_0 - 255 - 1 + 256 - 1) // 256 + 1) & (i >= 0) & (ninetoothed_tensor_0_index_0 < ((ninetoothed_ninetoothed_tensor_0_size_0 - 255 - 1 + 256 - 1) // 256 + 1 - ((ninetoothed_ninetoothed_tensor_0_size_0 - 255 - 1 + 256 - 1) // 256 + 1 - 1) - 1 + ((ninetoothed_ninetoothed_tensor_0_size_0 - 255 - 1 + 256 - 1) // 256 + 1) - 1) // ((ninetoothed_ninetoothed_tensor_0_size_0 - 255 - 1 + 256 - 1) // 256 + 1) + 1) & (ninetoothed_tensor_0_index_0 >= 0) & (i < (ninetoothed_ninetoothed_tensor_0_size_0 - 255 - 1 + 256 - 1) // 256 + 1) & (i >= 0) & (ninetoothed_tensor_0_index_0 * ((ninetoothed_ninetoothed_tensor_0_size_0 - 255 - 1 + 256 - 1) // 256 + 1) + i < ((ninetoothed_ninetoothed_tensor_0_size_0 - 255 - 1 + 256 - 1) // 256 + 1)) & (ninetoothed_tensor_0_index_0 * ((ninetoothed_ninetoothed_tensor_0_size_0 - 255 - 1 + 256 - 1) // 256 + 1) + i >= 0) & (triton.language.arange(0, 256)[::,] < 256) & (triton.language.arange(0, 256)[::,] >= 0) & ((ninetoothed_tensor_0_index_0 * ((ninetoothed_ninetoothed_tensor_0_size_0 - 255 - 1 + 256 - 1) // 256 + 1) + i) * 256 + triton.language.arange(0, 256)[::,] < ninetoothed_ninetoothed_tensor_0_size_0) & ((ninetoothed_tensor_0_index_0 * ((ninetoothed_ninetoothed_tensor_0_size_0 - 255 - 1 + 256 - 1) // 256 + 1) + i) * 256 + triton.language.arange(0, 256)[::,] >= 0) & ((ninetoothed_tensor_0_index_0 * ((ninetoothed_ninetoothed_tensor_0_size_0 - 255 - 1 + 256 - 1) // 256 + 1) + i) * 256 + triton.language.arange(0, 256)[::,] < ninetoothed_ninetoothed_tensor_0_size_0) & ((ninetoothed_tensor_0_index_0 * ((ninetoothed_ninetoothed_tensor_0_size_0 - 255 - 1 + 256 - 1) // 256 + 1) + i) * 256 + triton.language.arange(0, 256)[::,] >= 0) & ((ninetoothed_tensor_0_index_0 * ((ninetoothed_ninetoothed_tensor_0_size_0 - 255 - 1 + 256 - 1) // 256 + 1) + i) * 256 + triton.language.arange(0, 256)[::,] < ninetoothed_ninetoothed_tensor_0_size_0) & ((ninetoothed_tensor_0_index_0 * ((ninetoothed_ninetoothed_tensor_0_size_0 - 255 - 1 + 256 - 1) // 256 + 1) + i) * 256 + triton.language.arange(0, 256)[::,] >= 0), other=None), dtype)
        '\n    高精度数值稳定的反正切计算。\n    使用两级范围归约策略将输入映射到小区间，以保证多项式精度。\n    '
        ninetoothed_temporary_0_calc_dtype = dtype if dtype != triton.language.float16 else triton.language.float32
        ninetoothed_temporary_0_PI_OVER_2 = 1.5707963267948966
        ninetoothed_temporary_0_PI_OVER_4 = 0.7853981633974483
        ninetoothed_temporary_0_TAN_PI_8 = 0.414213562373095
        ninetoothed_temporary_0_x_arg = triton.language.cast(input_block, ninetoothed_temporary_0_calc_dtype)
        ninetoothed_temporary_0_sign = triton.language.where(ninetoothed_temporary_0_x_arg < 0.0, -1.0, 1.0)
        ninetoothed_temporary_0_abs_x = triton.language.abs(ninetoothed_temporary_0_x_arg)
        ninetoothed_temporary_0_mask_gt_1 = ninetoothed_temporary_0_abs_x > 1.0
        ninetoothed_temporary_0_safe_abs_x = triton.language.where(ninetoothed_temporary_0_mask_gt_1, ninetoothed_temporary_0_abs_x, triton.language.cast(1.0, ninetoothed_temporary_0_calc_dtype))
        ninetoothed_temporary_0_val_1 = triton.language.where(ninetoothed_temporary_0_mask_gt_1, triton.language.cast(1.0, ninetoothed_temporary_0_calc_dtype) / ninetoothed_temporary_0_safe_abs_x, ninetoothed_temporary_0_abs_x)
        ninetoothed_temporary_0_offset_1 = triton.language.where(ninetoothed_temporary_0_mask_gt_1, ninetoothed_temporary_0_PI_OVER_2, 0.0)
        ninetoothed_temporary_0_coef_1 = triton.language.where(ninetoothed_temporary_0_mask_gt_1, -1.0, 1.0)
        ninetoothed_temporary_0_mask_gt_tan_pi_8 = ninetoothed_temporary_0_val_1 > ninetoothed_temporary_0_TAN_PI_8
        ninetoothed_temporary_0_reduced_val = (ninetoothed_temporary_0_val_1 - 1.0) / (ninetoothed_temporary_0_val_1 + 1.0)
        ninetoothed_temporary_0_val_2 = triton.language.where(ninetoothed_temporary_0_mask_gt_tan_pi_8, ninetoothed_temporary_0_reduced_val, ninetoothed_temporary_0_val_1)
        ninetoothed_temporary_0_offset_2 = triton.language.where(ninetoothed_temporary_0_mask_gt_tan_pi_8, ninetoothed_temporary_0_PI_OVER_4, 0.0)
        '\n    计算 atan(x) 的泰勒级数近似。\n    有效范围：|x| <= 0.42\n    在此范围内，15阶多项式足以提供 float32/double 级别的精度。\n    '
        ninetoothed_temporary_0_x2 = ninetoothed_temporary_0_val_2 * ninetoothed_temporary_0_val_2
        ninetoothed_temporary_0_c3 = -0.333333333333
        ninetoothed_temporary_0_c5 = 0.2
        ninetoothed_temporary_0_c7 = -0.142857142857
        ninetoothed_temporary_0_c9 = 0.111111111111
        ninetoothed_temporary_0_c11 = -0.090909090909
        ninetoothed_temporary_0_c13 = 0.076923076923
        ninetoothed_temporary_0_c15 = -0.066666666667
        ninetoothed_temporary_0_p = triton.language.cast(ninetoothed_temporary_0_c15, ninetoothed_temporary_0_calc_dtype)
        ninetoothed_temporary_0_p = ninetoothed_temporary_0_p * ninetoothed_temporary_0_x2 + triton.language.cast(ninetoothed_temporary_0_c13, ninetoothed_temporary_0_calc_dtype)
        ninetoothed_temporary_0_p = ninetoothed_temporary_0_p * ninetoothed_temporary_0_x2 + triton.language.cast(ninetoothed_temporary_0_c11, ninetoothed_temporary_0_calc_dtype)
        ninetoothed_temporary_0_p = ninetoothed_temporary_0_p * ninetoothed_temporary_0_x2 + triton.language.cast(ninetoothed_temporary_0_c9, ninetoothed_temporary_0_calc_dtype)
        ninetoothed_temporary_0_p = ninetoothed_temporary_0_p * ninetoothed_temporary_0_x2 + triton.language.cast(ninetoothed_temporary_0_c7, ninetoothed_temporary_0_calc_dtype)
        ninetoothed_temporary_0_p = ninetoothed_temporary_0_p * ninetoothed_temporary_0_x2 + triton.language.cast(ninetoothed_temporary_0_c5, ninetoothed_temporary_0_calc_dtype)
        ninetoothed_temporary_0_p = ninetoothed_temporary_0_p * ninetoothed_temporary_0_x2 + triton.language.cast(ninetoothed_temporary_0_c3, ninetoothed_temporary_0_calc_dtype)
        ninetoothed_temporary_1_ = ninetoothed_temporary_0_val_2 + ninetoothed_temporary_0_val_2 * ninetoothed_temporary_0_x2 * ninetoothed_temporary_0_p
        ninetoothed_temporary_0_poly_res = ninetoothed_temporary_1_
        ninetoothed_temporary_0_atan_val_1 = triton.language.cast(ninetoothed_temporary_0_offset_2, ninetoothed_temporary_0_calc_dtype) + ninetoothed_temporary_0_poly_res
        ninetoothed_temporary_0_abs_result = triton.language.cast(ninetoothed_temporary_0_offset_1, ninetoothed_temporary_0_calc_dtype) + triton.language.cast(ninetoothed_temporary_0_coef_1, ninetoothed_temporary_0_calc_dtype) * ninetoothed_temporary_0_atan_val_1
        ninetoothed_temporary_0_final_result = triton.language.cast(ninetoothed_temporary_0_sign, ninetoothed_temporary_0_calc_dtype) * ninetoothed_temporary_0_abs_result
        ninetoothed_temporary_1_ = triton.language.cast(ninetoothed_temporary_0_final_result, dtype)
        result = ninetoothed_temporary_1_
        triton.language.store(ninetoothed_tensor_1_pointers + ((ninetoothed_tensor_1_index_0 * ((ninetoothed_ninetoothed_tensor_1_size_0 - 255 - 1 + 256 - 1) // 256 + 1) + i) * 256 + triton.language.arange(0, 256)[::,]) * ninetoothed_ninetoothed_tensor_1_stride_0, result, mask=True & (triton.language.arange(0, 256)[::,] < 256) & (triton.language.arange(0, 256)[::,] >= 0) & (i < (ninetoothed_ninetoothed_tensor_1_size_0 - 255 - 1 + 256 - 1) // 256 + 1) & (i >= 0) & (ninetoothed_tensor_1_index_0 < ((ninetoothed_ninetoothed_tensor_1_size_0 - 255 - 1 + 256 - 1) // 256 + 1 - ((ninetoothed_ninetoothed_tensor_1_size_0 - 255 - 1 + 256 - 1) // 256 + 1 - 1) - 1 + ((ninetoothed_ninetoothed_tensor_1_size_0 - 255 - 1 + 256 - 1) // 256 + 1) - 1) // ((ninetoothed_ninetoothed_tensor_1_size_0 - 255 - 1 + 256 - 1) // 256 + 1) + 1) & (ninetoothed_tensor_1_index_0 >= 0) & (i < (ninetoothed_ninetoothed_tensor_1_size_0 - 255 - 1 + 256 - 1) // 256 + 1) & (i >= 0) & (ninetoothed_tensor_1_index_0 * ((ninetoothed_ninetoothed_tensor_1_size_0 - 255 - 1 + 256 - 1) // 256 + 1) + i < ((ninetoothed_ninetoothed_tensor_1_size_0 - 255 - 1 + 256 - 1) // 256 + 1)) & (ninetoothed_tensor_1_index_0 * ((ninetoothed_ninetoothed_tensor_1_size_0 - 255 - 1 + 256 - 1) // 256 + 1) + i >= 0) & (triton.language.arange(0, 256)[::,] < 256) & (triton.language.arange(0, 256)[::,] >= 0) & ((ninetoothed_tensor_1_index_0 * ((ninetoothed_ninetoothed_tensor_1_size_0 - 255 - 1 + 256 - 1) // 256 + 1) + i) * 256 + triton.language.arange(0, 256)[::,] < ninetoothed_ninetoothed_tensor_1_size_0) & ((ninetoothed_tensor_1_index_0 * ((ninetoothed_ninetoothed_tensor_1_size_0 - 255 - 1 + 256 - 1) // 256 + 1) + i) * 256 + triton.language.arange(0, 256)[::,] >= 0) & ((ninetoothed_tensor_1_index_0 * ((ninetoothed_ninetoothed_tensor_1_size_0 - 255 - 1 + 256 - 1) // 256 + 1) + i) * 256 + triton.language.arange(0, 256)[::,] < ninetoothed_ninetoothed_tensor_1_size_0) & ((ninetoothed_tensor_1_index_0 * ((ninetoothed_ninetoothed_tensor_1_size_0 - 255 - 1 + 256 - 1) // 256 + 1) + i) * 256 + triton.language.arange(0, 256)[::,] >= 0) & ((ninetoothed_tensor_1_index_0 * ((ninetoothed_ninetoothed_tensor_1_size_0 - 255 - 1 + 256 - 1) // 256 + 1) + i) * 256 + triton.language.arange(0, 256)[::,] < ninetoothed_ninetoothed_tensor_1_size_0) & ((ninetoothed_tensor_1_index_0 * ((ninetoothed_ninetoothed_tensor_1_size_0 - 255 - 1 + 256 - 1) // 256 + 1) + i) * 256 + triton.language.arange(0, 256)[::,] >= 0))

def launch_application(ninetoothed_tensor_0, ninetoothed_tensor_1):
    application[lambda meta: (((ninetoothed_tensor_0.size(0) - 255 - 1 + 256 - 1) // 256 + 1 - ((ninetoothed_tensor_0.size(0) - 255 - 1 + 256 - 1) // 256 + 1 - 1) - 1 + ((ninetoothed_tensor_0.size(0) - 255 - 1 + 256 - 1) // 256 + 1) - 1) // ((ninetoothed_tensor_0.size(0) - 255 - 1 + 256 - 1) // 256 + 1) + 1,)](ninetoothed_tensor_0, ninetoothed_tensor_0.size(0), ninetoothed_tensor_0.stride(0), ninetoothed_tensor_1, ninetoothed_tensor_1.size(0), ninetoothed_tensor_1.stride(0), num_warps=8, num_stages=7)
