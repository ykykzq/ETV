import triton
import triton.language

@triton.jit
def application_all_elements(ninetoothed_ninetoothed_tensor_0_pointer, ninetoothed_ninetoothed_tensor_0_size_0, ninetoothed_ninetoothed_tensor_0_stride_0, ninetoothed_ninetoothed_tensor_1_pointer, ninetoothed_ninetoothed_tensor_1_size_0, ninetoothed_ninetoothed_tensor_1_stride_0):
    ninetoothed_pid = triton.language.program_id(0)
    ninetoothed_tensor_0_index_0 = ninetoothed_pid
    ninetoothed_tensor_0_pointers = ninetoothed_ninetoothed_tensor_0_pointer
    ninetoothed_tensor_1_index_0 = ninetoothed_pid
    ninetoothed_tensor_1_pointers = ninetoothed_ninetoothed_tensor_1_pointer
    triton.language.store(ninetoothed_tensor_1_pointers + ninetoothed_tensor_1_index_0 * ninetoothed_ninetoothed_tensor_1_stride_0, triton.language.sum(triton.language.load(ninetoothed_tensor_0_pointers + (ninetoothed_tensor_0_index_0 * 1024 + triton.language.arange(0, 1024)[::,]) * ninetoothed_ninetoothed_tensor_0_stride_0, mask=True & (ninetoothed_tensor_0_index_0 < (ninetoothed_ninetoothed_tensor_0_size_0 - 1023 - 1 + 1024 - 1) // 1024 + 1) & (ninetoothed_tensor_0_index_0 >= 0) & (triton.language.arange(0, 1024)[::,] < 1024) & (triton.language.arange(0, 1024)[::,] >= 0) & (ninetoothed_tensor_0_index_0 * 1024 + triton.language.arange(0, 1024)[::,] < ninetoothed_ninetoothed_tensor_0_size_0) & (ninetoothed_tensor_0_index_0 * 1024 + triton.language.arange(0, 1024)[::,] >= 0) & (ninetoothed_tensor_0_index_0 * 1024 + triton.language.arange(0, 1024)[::,] < ninetoothed_ninetoothed_tensor_0_size_0) & (ninetoothed_tensor_0_index_0 * 1024 + triton.language.arange(0, 1024)[::,] >= 0), other=None), 0), mask=True & (ninetoothed_tensor_1_index_0 < ninetoothed_ninetoothed_tensor_1_size_0 - 1 + 1 - 1 + 1) & (ninetoothed_tensor_1_index_0 >= 0) & (0 < 1) & (0 >= 0) & (ninetoothed_tensor_1_index_0 < ninetoothed_ninetoothed_tensor_1_size_0) & (ninetoothed_tensor_1_index_0 >= 0))

def launch_application_all_elements(ninetoothed_tensor_0, ninetoothed_tensor_1):
    application_all_elements[lambda meta: ((ninetoothed_tensor_0.size(0) - 1023 - 1 + 1024 - 1) // 1024 + 1,)](ninetoothed_tensor_0, ninetoothed_tensor_0.size(0), ninetoothed_tensor_0.stride(0), ninetoothed_tensor_1, ninetoothed_tensor_1.size(0), ninetoothed_tensor_1.stride(0), num_warps=8, num_stages=7)
