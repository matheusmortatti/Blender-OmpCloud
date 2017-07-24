/*
 * ***** BEGIN GPL LICENSE BLOCK *****
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version. 
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 *
 * The Original Code is Copyright (C) 2006 Blender Foundation.
 * All rights reserved.
 *
 * The Original Code is: all of this file.
 *
 * Contributor(s): none yet.
 *
 * ***** END GPL LICENSE BLOCK *****
 */

/** \file blender/nodes/composite/nodes/node_composite_value.c
 *  \ingroup cmpnodes
 */


#include "node_composite_util.h"

/* **************** VALUE ******************** */
static bNodeSocketTemplate cmp_node_value_out[] = {
	{	SOCK_FLOAT, 0, N_("Value"), 0.5f, 0, 0, 0, -FLT_MAX, FLT_MAX, PROP_NONE},
	{	-1, 0, ""	}
};

void register_node_type_cmp_value(void)
{
	static bNodeType ntype;

	cmp_node_type_base(&ntype, CMP_NODE_VALUE, "Value", NODE_CLASS_INPUT, 0);
	node_type_socket_templates(&ntype, NULL, cmp_node_value_out);
	node_type_size_preset(&ntype, NODE_SIZE_SMALL);

	nodeRegisterType(&ntype);
}
