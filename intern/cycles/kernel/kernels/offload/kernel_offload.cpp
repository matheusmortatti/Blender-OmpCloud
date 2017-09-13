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

/*
 * The offload mode was tested with this settings:
 * two KNC devices (2x Intel Xeon Phi 7120P), Linux, Intel compiler 2016.03.
 *
 * The KNC does not support SSE/AVX optimization. We disable it with #define __KERNEL_OFFLOAD__.
 *
 * We are not using the automatic Mapped variable method of offload mode (mapping of a variable
 * in a data environment to a variable in a device data environment),
 * because it is unstable in some cases:
 *
 * #define ALLOC alloc_if(1) free_if(0)
 * #define FREE alloc_if(0) free_if(1)
 * #define REUSE alloc_if(0) free_if(0)
 * #define ONE_USE
 *
 */

#define __KERNEL_OFFLOAD__

//#include "util_debug.h"

#include "kernel_offload.h"

#pragma offload_attribute(push, target(mic))
#	include "kernel/kernel.h"
#	define KERNEL_ARCH offload
#	include "../cpu/kernel_cpu_impl.h"
#pragma offload_attribute(pop)

#include <omp.h>

CCL_NAMESPACE_BEGIN

//DebugFlags DebugFlags::instance;

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
		int tile_w)
{
//#pragma omp target //offload target(mic:numDevice)
	{
		int tile_size = tile_h*tile_w;

#pragma omp parallel for schedule(dynamic, 1)
		for (int i = 0; i < tile_size; i++) {
			int y = i / tile_w;
			int x = i - y * tile_w;
		for (int sample = start_sample; sample < end_sample; sample++)
			kernel_offload_path_trace((KernelGlobals *) kg_bin,
					(float *) buffer_bin, (unsigned int*) rng_state_bin,
					sample, x + tile_x, y + tile_y, offset, stride);
		}
	}
}

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
		int task_w)
{
//#pragma omp target //offload target(mic:numDevice)
	{
		int task_size = task_h*task_w;

#pragma omp parallel for schedule(dynamic, 1)
		for (int i = 0; i < task_size; i++) {
			int y = i / task_w;
			int x = i - y * task_w;

			kernel_offload_convert_to_half_float((KernelGlobals *) kg_bin,
					(uchar4*) rgba_half_bin, (float*) buffer_bin,
					sample_scale, x + task_x, y + task_y, task_offset, task_stride);
		}
	}
}

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
		int task_w)
{
//#pragma omp target //offload target(mic:numDevice)
	{
		int task_size = task_h*task_w;

#pragma omp parallel for schedule(dynamic, 1)
		for (int i = 0; i < task_size; i++) {
			int y = i / task_w;
			int x = i - y * task_w;

			kernel_offload_convert_to_byte((KernelGlobals *) kg_bin,
					(uchar4*) rgba_byte_bin, (float*) buffer_bin,
					sample_scale, x + task_x, y + task_y, task_offset, task_stride);
		}
	}
}

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
		int sample)
{
//#pragma omp target //offload target(mic:numDevice)
	{
#pragma omp parallel for schedule(dynamic, 1)
		for (int x = task_shader_x; x < task_shader_x + task_shader_w; x++)
			kernel_offload_shader((KernelGlobals *) kg_bin,
				(uint4*) shader_input_bin, (float4*) shader_output_bin,
				(float*) shader_output_luma, task_shader_eval_type,
				task_shader_filter, x, task_offset, sample);
	}
}

DEVICE_PTR offload_alloc_kg(int numDevice)
{
	DEVICE_PTR kg_bin = NULL;

//#pragma omp target map(from: kg_bin) //offload target(mic:numDevice) out(kg_bin)
	{
		KernelGlobals *kg = new KernelGlobals();
		kg_bin = (DEVICE_PTR) kg;
	}

	return (DEVICE_PTR) kg_bin;
}

void offload_free_kg(int numDevice, DEVICE_PTR kg_bin)
{
//#pragma omp target map(to: kg_bin) //offload target(mic:numDevice) in(kg_bin)
	{
		KernelGlobals *kg = (KernelGlobals *) kg_bin;
		delete kg;
	}
}

DEVICE_PTR offload_mem_alloc(int numDevice, DEVICE_PTR mem, SIZE_T memSize)
{
	DEVICE_PTR mem_device;

//#pragma omp target map(from: mem_device) //offload target(mic:numDevice) out(mem_device)
	{
		mem_device = (DEVICE_PTR) new char[memSize];
	}

	return mem_device;
}

void offload_mem_copy_to(int numDevice, char *memh, DEVICE_PTR mem, SIZE_T memSize)
{
//#pragma omp target map(to: memh[0:memSize-1]) //offload target(mic:numDevice) in(memh:length(memSize))
	{
		memcpy((char*) mem, memh, memSize);
	}
}

void offload_mem_copy_from(int numDevice, DEVICE_PTR mem, char *memh, SIZE_T offset, SIZE_T memSize)
{
	char *temp = new char[memSize];

//#pragma omp target map(from: temp[0:memSize-1]) //offload target(mic:numDevice) out(temp:length(memSize))
	{
		memcpy(temp, (char*) mem + offset, memSize);
	}

	memcpy(memh + offset, temp, memSize);
	delete[]temp;
}

void offload_mem_zero(int numDevice, DEVICE_PTR mem, SIZE_T memSize)
{
//#pragma omp target //offload target(mic:numDevice)
	{
		memset((char*) mem, 0, memSize);
	}
}

void offload_mem_free(int numDevice, DEVICE_PTR mem, SIZE_T memSize)
{
//#pragma omp target //offload target(mic:numDevice)
	{
		delete[](char*) mem;
	}
}

void offload_const_copy(int numDevice, DEVICE_PTR kg_bin, const char *name, char *host_bin, SIZE_T size)
{
	if (strcmp(name, "__data") == 0) {
//#pragma omp target map(to: host_bin[0:size-1]) //offload target(mic:numDevice) in(host_bin:length(size))
		{
			KernelGlobals *kg = (KernelGlobals *) kg_bin;
			memcpy(&kg->__data, host_bin, size);
		}
	}
}

//#pragma offload_attribute(push, target(mic))
void offload_kernel_tex_copy_internal(KernelGlobals *kg,
		const char *name,
		device_ptr mem,
		size_t width,
		size_t height,
		size_t depth,
		InterpolationType interpolation,
		ExtensionType extension)
{
	if (0) {
	}

#define KERNEL_TEX(type, ttype, tname) \
	else if(strcmp(name, #tname) == 0) { \
		kg->tname.data = (type*)mem; \
		kg->tname.width = width; \
	}
#define KERNEL_IMAGE_TEX(type, ttype, tname)
#include "kernel/kernel_textures.h"

	else if (strstr(name, "__tex_image_float4")) {
		texture_image_float4 *tex = NULL;
		int id = atoi(name + strlen("__tex_image_float4_"));
		int array_index = id;

		if (array_index >= 0 && array_index < TEX_NUM_FLOAT4_CPU) {
			tex = &kg->texture_float4_images[array_index];
		}

		if (tex) {
			tex->data = (float4*)mem;
			tex->dimensions_set(width, height, depth);
			tex->interpolation = interpolation;
			tex->extension = extension;
		}
	}
	else if (strstr(name, "__tex_image_float")) {
		texture_image_float *tex = NULL;
		int id = atoi(name + strlen("__tex_image_float_"));
		int array_index = id - TEX_START_FLOAT_CPU;

		if (array_index >= 0 && array_index < TEX_NUM_FLOAT_CPU) {
			tex = &kg->texture_float_images[array_index];
		}

		if (tex) {
			tex->data = (float*)mem;
			tex->dimensions_set(width, height, depth);
			tex->interpolation = interpolation;
			tex->extension = extension;
		}
	}
	else if (strstr(name, "__tex_image_byte4")) {
		texture_image_uchar4 *tex = NULL;
		int id = atoi(name + strlen("__tex_image_byte4_"));
		int array_index = id - TEX_START_BYTE4_CPU;

		if (array_index >= 0 && array_index < TEX_NUM_BYTE4_CPU) {
			tex = &kg->texture_byte4_images[array_index];
		}

		if (tex) {
			tex->data = (uchar4*)mem;
			tex->dimensions_set(width, height, depth);
			tex->interpolation = interpolation;
			tex->extension = extension;
		}
	}
	else if (strstr(name, "__tex_image_byte")) {
		texture_image_uchar *tex = NULL;
		int id = atoi(name + strlen("__tex_image_byte_"));
		int array_index = id - TEX_START_BYTE_CPU;

		if (array_index >= 0 && array_index < TEX_NUM_BYTE_CPU) {
			tex = &kg->texture_byte_images[array_index];
		}

		if (tex) {
			tex->data = (uchar*)mem;
			tex->dimensions_set(width, height, depth);
			tex->interpolation = interpolation;
			tex->extension = extension;
		}
	}
	else if (strstr(name, "__tex_image_half4")) {
		texture_image_half4 *tex = NULL;
		int id = atoi(name + strlen("__tex_image_half4_"));
		int array_index = id - TEX_START_HALF4_CPU;

		if (array_index >= 0 && array_index < TEX_NUM_HALF4_CPU) {
			tex = &kg->texture_half4_images[array_index];
		}

		if (tex) {
			tex->data = (half4*)mem;
			tex->dimensions_set(width, height, depth);
			tex->interpolation = interpolation;
			tex->extension = extension;
		}
	}
	else if (strstr(name, "__tex_image_half")) {
		texture_image_half *tex = NULL;
		int id = atoi(name + strlen("__tex_image_half_"));
		int array_index = id - TEX_START_HALF_CPU;

		if (array_index >= 0 && array_index < TEX_NUM_HALF_CPU) {
			tex = &kg->texture_half_images[array_index];
		}

		if (tex) {
			tex->data = (half*)mem;
			tex->dimensions_set(width, height, depth);
			tex->interpolation = interpolation;
			tex->extension = extension;
		}
	}
	else
		kernel_assert(0);
}
//#pragma offload_attribute(pop)

DEVICE_PTR offload_tex_copy(int numDevice,
		DEVICE_PTR kg_bin,
		const char *name_bin,
		char* mem,
		SIZE_T size,
		SIZE_T width,
		SIZE_T height,
		SIZE_T depth,
		int interpolation,
		int extension)
{
	if (name_bin == NULL || mem == NULL)
		return NULL;

	SIZE_T nameSize = sizeof (char) * (strlen(name_bin) + 1);
	char *name = (char *) name_bin;

	DEVICE_PTR tex_mem_device = NULL;

//#pragma omp target \ //offload target(mic:numDevice) \
            map(to: mem[0:size-1], name[0:nameSize-1]) \//in(mem:length(size)) \
            //in(name:length(nameSize)) \
            map(from: tex_mem_device )//out(tex_mem_device)
	{
		char* tex_mem_bin = new char[size];
		memcpy(tex_mem_bin, mem, size);

		offload_kernel_tex_copy_internal((KernelGlobals*) kg_bin, name,
				(DEVICE_PTR) tex_mem_bin, width, height, depth,
				(InterpolationType) interpolation, (ExtensionType) extension);

		tex_mem_device = (DEVICE_PTR) tex_mem_bin;
	}

	return tex_mem_device;
}

void offload_tex_free(int numDevice, DEVICE_PTR kg_bin, DEVICE_PTR mem, SIZE_T memSize)
{
	offload_mem_free(numDevice, mem, memSize);
}

void offload_kernel_globals_init(int numDevice, DEVICE_PTR kg_bin)
{
//#pragma omp target map(to: kg_bin) //offload target(mic:numDevice) in(kg_bin)
	{
		KernelGlobals *kg = (KernelGlobals *) kg_bin;

		kg->transparent_shadow_intersections = NULL;
		const int decoupled_count = sizeof (kg->decoupled_volume_steps) /
				sizeof (*kg->decoupled_volume_steps);
		for (int i = 0; i < decoupled_count; ++i) {
			kg->decoupled_volume_steps[i] = NULL;
		}
		kg->decoupled_volume_steps_index = 0;
	}
}

void offload_kernel_globals_free(int numDevice, DEVICE_PTR kg_bin)
{
//#pragma omp target map(to: kg_bin) //offload target(mic:numDevice) in(kg_bin)
	{
		KernelGlobals *kg = (KernelGlobals *) kg_bin;
		if (kg->transparent_shadow_intersections != NULL) {
			free(kg->transparent_shadow_intersections);
		}
		const int decoupled_count = sizeof (kg->decoupled_volume_steps) /
				sizeof (*kg->decoupled_volume_steps);
		for (int i = 0; i < decoupled_count; ++i) {
			if (kg->decoupled_volume_steps[i] != NULL) {
				free(kg->decoupled_volume_steps[i]);
			}
		}
	}
}

int offload_devices()
{
#if _OPENMP >= 201307
	return 1; //omp_get_num_devices();
#else
	const char *num = getenv("OMP_GET_NUM_DEVICES");
	if (num) {
		return atoi(num);
	}
	else {
		return 0;
	}
#endif
}

CCL_NAMESPACE_END