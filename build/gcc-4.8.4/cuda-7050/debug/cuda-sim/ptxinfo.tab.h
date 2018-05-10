
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
     INT_OPERAND = 258,
     HEADER = 259,
     INFO = 260,
     FUNC = 261,
     USED = 262,
     REGS = 263,
     BYTES = 264,
     LMEM = 265,
     SMEM = 266,
     CMEM = 267,
     GMEM = 268,
     IDENTIFIER = 269,
     PLUS = 270,
     COMMA = 271,
     LEFT_SQUARE_BRACKET = 272,
     RIGHT_SQUARE_BRACKET = 273,
     COLON = 274,
     SEMICOLON = 275,
     QUOTE = 276,
     LINE = 277,
     WARNING = 278,
     FOR = 279,
     TEXTURES = 280,
     DUPLICATE = 281,
     FUNCTION = 282,
     VARIABLE = 283,
     FATAL = 284
   };
#endif



#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union YYSTYPE
{

/* Line 1676 of yacc.c  */
#line 30 "ptxinfo.y"

  int    int_value;
  char * string_value;



/* Line 1676 of yacc.c  */
#line 88 "/home/j/code/gpgpu-sim_distribution-dev/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptxinfo.tab.h"
} YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
#endif

extern YYSTYPE ptxinfo_lval;


