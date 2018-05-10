
/* A Bison parser, made by GNU Bison 2.4.1.  */

/* Skeleton interface for Bison's Yacc-like parsers in C
   
      Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002, 2003, 2004, 2005, 2006
   Free Software Foundation, Inc.
   
   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
   
   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.
   
   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */


/* Tokens.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
   /* Put the tokens into the symbol table, so that GDB and other debuggers
      know about them.  */
   enum yytokentype {
     STRING = 258,
     OPCODE = 259,
     ALIGN_DIRECTIVE = 260,
     BRANCHTARGETS_DIRECTIVE = 261,
     BYTE_DIRECTIVE = 262,
     CALLPROTOTYPE_DIRECTIVE = 263,
     CALLTARGETS_DIRECTIVE = 264,
     CONST_DIRECTIVE = 265,
     CONSTPTR_DIRECTIVE = 266,
     PTR_DIRECTIVE = 267,
     ENTRY_DIRECTIVE = 268,
     EXTERN_DIRECTIVE = 269,
     FILE_DIRECTIVE = 270,
     FUNC_DIRECTIVE = 271,
     GLOBAL_DIRECTIVE = 272,
     LOCAL_DIRECTIVE = 273,
     LOC_DIRECTIVE = 274,
     MAXNCTAPERSM_DIRECTIVE = 275,
     MAXNNREG_DIRECTIVE = 276,
     MAXNTID_DIRECTIVE = 277,
     MINNCTAPERSM_DIRECTIVE = 278,
     PARAM_DIRECTIVE = 279,
     PRAGMA_DIRECTIVE = 280,
     REG_DIRECTIVE = 281,
     REQNTID_DIRECTIVE = 282,
     SECTION_DIRECTIVE = 283,
     SHARED_DIRECTIVE = 284,
     SREG_DIRECTIVE = 285,
     STRUCT_DIRECTIVE = 286,
     SURF_DIRECTIVE = 287,
     TARGET_DIRECTIVE = 288,
     TEX_DIRECTIVE = 289,
     UNION_DIRECTIVE = 290,
     VERSION_DIRECTIVE = 291,
     ADDRESS_SIZE_DIRECTIVE = 292,
     VISIBLE_DIRECTIVE = 293,
     WEAK_DIRECTIVE = 294,
     IDENTIFIER = 295,
     INT_OPERAND = 296,
     FLOAT_OPERAND = 297,
     DOUBLE_OPERAND = 298,
     S8_TYPE = 299,
     S16_TYPE = 300,
     S32_TYPE = 301,
     S64_TYPE = 302,
     U8_TYPE = 303,
     U16_TYPE = 304,
     U32_TYPE = 305,
     U64_TYPE = 306,
     F16_TYPE = 307,
     F32_TYPE = 308,
     F64_TYPE = 309,
     FF64_TYPE = 310,
     B8_TYPE = 311,
     B16_TYPE = 312,
     B32_TYPE = 313,
     B64_TYPE = 314,
     BB64_TYPE = 315,
     BB128_TYPE = 316,
     PRED_TYPE = 317,
     TEXREF_TYPE = 318,
     SAMPLERREF_TYPE = 319,
     SURFREF_TYPE = 320,
     V2_TYPE = 321,
     V3_TYPE = 322,
     V4_TYPE = 323,
     COMMA = 324,
     PRED = 325,
     HALF_OPTION = 326,
     EXTP_OPTION = 327,
     EQ_OPTION = 328,
     NE_OPTION = 329,
     LT_OPTION = 330,
     LE_OPTION = 331,
     GT_OPTION = 332,
     GE_OPTION = 333,
     LO_OPTION = 334,
     LS_OPTION = 335,
     HI_OPTION = 336,
     HS_OPTION = 337,
     EQU_OPTION = 338,
     NEU_OPTION = 339,
     LTU_OPTION = 340,
     LEU_OPTION = 341,
     GTU_OPTION = 342,
     GEU_OPTION = 343,
     NUM_OPTION = 344,
     NAN_OPTION = 345,
     CF_OPTION = 346,
     SF_OPTION = 347,
     NSF_OPTION = 348,
     LEFT_SQUARE_BRACKET = 349,
     RIGHT_SQUARE_BRACKET = 350,
     WIDE_OPTION = 351,
     SPECIAL_REGISTER = 352,
     MINUS = 353,
     PLUS = 354,
     COLON = 355,
     SEMI_COLON = 356,
     EXCLAMATION = 357,
     PIPE = 358,
     RIGHT_BRACE = 359,
     LEFT_BRACE = 360,
     EQUALS = 361,
     PERIOD = 362,
     BACKSLASH = 363,
     DIMENSION_MODIFIER = 364,
     RN_OPTION = 365,
     RZ_OPTION = 366,
     RM_OPTION = 367,
     RP_OPTION = 368,
     RNI_OPTION = 369,
     RZI_OPTION = 370,
     RMI_OPTION = 371,
     RPI_OPTION = 372,
     UNI_OPTION = 373,
     GEOM_MODIFIER_1D = 374,
     GEOM_MODIFIER_2D = 375,
     GEOM_MODIFIER_3D = 376,
     SAT_OPTION = 377,
     FTZ_OPTION = 378,
     NEG_OPTION = 379,
     SYNC_OPTION = 380,
     RED_OPTION = 381,
     ARRIVE_OPTION = 382,
     ATOMIC_POPC = 383,
     ATOMIC_AND = 384,
     ATOMIC_OR = 385,
     ATOMIC_XOR = 386,
     ATOMIC_CAS = 387,
     ATOMIC_EXCH = 388,
     ATOMIC_ADD = 389,
     ATOMIC_INC = 390,
     ATOMIC_DEC = 391,
     ATOMIC_MIN = 392,
     ATOMIC_MAX = 393,
     LEFT_ANGLE_BRACKET = 394,
     RIGHT_ANGLE_BRACKET = 395,
     LEFT_PAREN = 396,
     RIGHT_PAREN = 397,
     APPROX_OPTION = 398,
     FULL_OPTION = 399,
     ANY_OPTION = 400,
     ALL_OPTION = 401,
     BALLOT_OPTION = 402,
     GLOBAL_OPTION = 403,
     CTA_OPTION = 404,
     SYS_OPTION = 405,
     EXIT_OPTION = 406,
     ABS_OPTION = 407,
     TO_OPTION = 408,
     CA_OPTION = 409,
     CG_OPTION = 410,
     CS_OPTION = 411,
     LU_OPTION = 412,
     CV_OPTION = 413,
     WB_OPTION = 414,
     WT_OPTION = 415,
     NC_OPTION = 416,
     UP_OPTION = 417,
     DOWN_OPTION = 418,
     BFLY_OPTION = 419,
     IDX_OPTION = 420
   };
#endif



#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union YYSTYPE
{

/* Line 1676 of yacc.c  */
#line 30 "../src/cuda-sim/ptx.y"

  double double_value;
  float  float_value;
  int    int_value;
  char * string_value;
  void * ptr_value;



/* Line 1676 of yacc.c  */
#line 227 "/home/j/code/gpgpu-sim_distribution-dev/build/gcc-4.8.4/cuda-7050/debug/cuobjdump_to_ptxplus/ptx.tab.h"
} YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
#endif

extern YYSTYPE ptx_lval;


