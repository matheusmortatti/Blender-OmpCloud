/*
 * Copyright 2011-2013 Blender Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Contributors: 2016 Milan Jaros (milan.jaros@vsb.cz)
 *
 */

#ifndef __KERNEL_OFFLOAD_H__
#define __KERNEL_OFFLOAD_H__

#define DEVICE_PTR unsigned long long
#define SIZE_T long

CCL_NAMESPACE_BEGIN

void offload_path_trace(int numDevice,
	DEVICE_PTR kg_bin,
	DEVICE_PTR buffer_bin,
	DEVICE_PTR rng_state_bin,
	int start_sample,int end_sample,
	int tile_x,
	int tile_y,
	int offset,
	int stride,
	int tile_h,
	int tile_w);

void offload_convert_to_half_float_kernel(int numDevice,
	DEVICE_PTR kg_bin,
	DEVICE_PTR rgba_half_bin,
	DEVICE_PTR buffer_bin,
	float sample_scale,
	int task_x,
	int task_y,
	int task_offset,
	int task_stride,
	int task_h,
	int task_w);

void offload_convert_to_byte_kernel(int numDevice,
	DEVICE_PTR kg_bin,
	DEVICE_PTR rgba_byte_bin,
	DEVICE_PTR buffer_bin,
	float sample_scale,
	int task_x,
	int task_y,
	int task_offset,
	int task_stride,
	int task_h,
	int task_w);

void offload_shader_kernel(int numDevice,
	DEVICE_PTR kg_bin,
	DEVICE_PTR shader_input_bin,
	DEVICE_PTR shader_output_bin,
	DEVICE_PTR shader_output_luma,
	int task_shader_eval_type,
	int task_shader_filter,
	int task_shader_x,
	int task_shader_w,
	int task_offset,
	int sample);

DEVICE_PTR offload_alloc_kg(int numDevice);
void offload_free_kg(int numDevice, DEVICE_PTR kg);

DEVICE_PTR offload_mem_alloc(int numDevice, DEVICE_PTR mem, SIZE_T memSize);
void offload_mem_copy_to(int numDevice, char *memh, DEVICE_PTR mem, SIZE_T memSize);
void offload_mem_copy_from(int numDevice, DEVICE_PTR mem, char *memh, SIZE_T offset, SIZE_T memSize);
void offload_mem_zero(int numDevice, DEVICE_PTR mem, SIZE_T memSize);
void offload_mem_free(int numDevice, DEVICE_PTR mem, SIZE_T memSize);

void offload_const_copy(int numDevice, DEVICE_PTR kg, const char *name, char *host, SIZE_T size);

DEVICE_PTR offload_tex_copy(int numDevice,
	DEVICE_PTR kg_bin,
	const char *name_bin,
	char* mem,
	SIZE_T size,
	SIZE_T width,
	SIZE_T height,
	SIZE_T depth,
	int interpolation,
	int extension);

void offload_tex_free(int numDevice, DEVICE_PTR kg_bin, DEVICE_PTR mem, SIZE_T memSize);

void offload_kernel_globals_init(int numDevice, DEVICE_PTR kernel_globals);
void offload_kernel_globals_free(int numDevice, DEVICE_PTR kernel_globals);

int offload_devices();

CCL_NAMESPACE_END

#endif /* __KERNEL_OFFLOAD_H__ */