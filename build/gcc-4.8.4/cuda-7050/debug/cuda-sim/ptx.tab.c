
/* A Bison parser, made by GNU Bison 2.4.1.  */

/* Skeleton implementation for Bison's Yacc-like parsers in C
   
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

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "2.4.1"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1

/* Using locations.  */
#define YYLSP_NEEDED 0

/* Substitute the variable and function names.  */
#define yyparse         ptx_parse
#define yylex           ptx_lex
#define yyerror         ptx_error
#define yylval          ptx_lval
#define yychar          ptx_char
#define yydebug         ptx_debug
#define yynerrs         ptx_nerrs


/* Copy the first part of user declarations.  */


/* Line 189 of yacc.c  */
#line 81 "/home/j/code/gpgpu-sim_distribution-dev/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c"

/* Enabling traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 0
#endif

/* Enabling the token table.  */
#ifndef YYTOKEN_TABLE
# define YYTOKEN_TABLE 0
#endif


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

/* Line 214 of yacc.c  */
#line 30 "ptx.y"

  double double_value;
  float  float_value;
  int    int_value;
  char * string_value;
  void * ptr_value;



/* Line 214 of yacc.c  */
#line 292 "/home/j/code/gpgpu-sim_distribution-dev/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c"
} YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
#endif


/* Copy the second part of user declarations.  */

/* Line 264 of yacc.c  */
#line 205 "ptx.y"

  	#include "ptx_parser.h"
	#include <stdlib.h>
	#include <string.h>
	#include <math.h>
	void syntax_not_implemented();
	extern int g_func_decl;
	int ptx_lex(void);
	int ptx_error(const char *);


/* Line 264 of yacc.c  */
#line 316 "/home/j/code/gpgpu-sim_distribution-dev/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c"

#ifdef short
# undef short
#endif

#ifdef YYTYPE_UINT8
typedef YYTYPE_UINT8 yytype_uint8;
#else
typedef unsigned char yytype_uint8;
#endif

#ifdef YYTYPE_INT8
typedef YYTYPE_INT8 yytype_int8;
#elif (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
typedef signed char yytype_int8;
#else
typedef short int yytype_int8;
#endif

#ifdef YYTYPE_UINT16
typedef YYTYPE_UINT16 yytype_uint16;
#else
typedef unsigned short int yytype_uint16;
#endif

#ifdef YYTYPE_INT16
typedef YYTYPE_INT16 yytype_int16;
#else
typedef short int yytype_int16;
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif ! defined YYSIZE_T && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned int
# endif
#endif

#define YYSIZE_MAXIMUM ((YYSIZE_T) -1)

#ifndef YY_
# if YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(msgid) dgettext ("bison-runtime", msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(msgid) msgid
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(e) ((void) (e))
#else
# define YYUSE(e) /* empty */
#endif

/* Identity function, used to suppress warnings about constant conditions.  */
#ifndef lint
# define YYID(n) (n)
#else
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static int
YYID (int yyi)
#else
static int
YYID (yyi)
    int yyi;
#endif
{
  return yyi;
}
#endif

#if ! defined yyoverflow || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#     ifndef _STDLIB_H
#      define _STDLIB_H 1
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's `empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (YYID (0))
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined _STDLIB_H \
       && ! ((defined YYMALLOC || defined malloc) \
	     && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef _STDLIB_H
#    define _STDLIB_H 1
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* ! defined yyoverflow || YYERROR_VERBOSE */


#if (! defined yyoverflow \
     && (! defined __cplusplus \
	 || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yytype_int16 yyss_alloc;
  YYSTYPE yyvs_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (yytype_int16) + sizeof (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

/* Copy COUNT objects from FROM to TO.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(To, From, Count) \
      __builtin_memcpy (To, From, (Count) * sizeof (*(From)))
#  else
#   define YYCOPY(To, From, Count)		\
      do					\
	{					\
	  YYSIZE_T yyi;				\
	  for (yyi = 0; yyi < (Count); yyi++)	\
	    (To)[yyi] = (From)[yyi];		\
	}					\
      while (YYID (0))
#  endif
# endif

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack_alloc, Stack)				\
    do									\
      {									\
	YYSIZE_T yynewbytes;						\
	YYCOPY (&yyptr->Stack_alloc, Stack, yysize);			\
	Stack = &yyptr->Stack_alloc;					\
	yynewbytes = yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
	yyptr += yynewbytes / sizeof (*yyptr);				\
      }									\
    while (YYID (0))

#endif

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  2
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   587

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  166
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  66
/* YYNRULES -- Number of rules.  */
#define YYNRULES  287
/* YYNRULES -- Number of states.  */
#define YYNSTATES  395

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   420

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,   106,   107,   108,   109,   110,   111,   112,   113,   114,
     115,   116,   117,   118,   119,   120,   121,   122,   123,   124,
     125,   126,   127,   128,   129,   130,   131,   132,   133,   134,
     135,   136,   137,   138,   139,   140,   141,   142,   143,   144,
     145,   146,   147,   148,   149,   150,   151,   152,   153,   154,
     155,   156,   157,   158,   159,   160,   161,   162,   163,   164,
     165
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     4,     7,    10,    13,    14,    18,    19,
      20,    26,    33,    36,    39,    41,    44,    45,    46,    54,
      55,    59,    61,    62,    63,    70,    72,    74,    77,    80,
      82,    85,    88,    91,    94,    95,    97,    98,   103,   104,
     110,   111,   116,   117,   121,   124,   126,   128,   130,   133,
     137,   139,   141,   144,   147,   148,   152,   153,   156,   159,
     162,   166,   169,   174,   181,   184,   188,   196,   201,   205,
     208,   211,   216,   221,   228,   230,   232,   236,   238,   243,
     247,   252,   254,   257,   259,   261,   263,   265,   267,   270,
     272,   274,   276,   278,   280,   282,   284,   286,   288,   290,
     292,   295,   297,   299,   301,   303,   305,   307,   309,   311,
     313,   315,   317,   319,   321,   323,   325,   327,   329,   331,
     333,   335,   337,   339,   341,   343,   345,   349,   353,   355,
     359,   362,   365,   369,   370,   382,   389,   395,   398,   400,
     401,   405,   407,   410,   414,   418,   422,   426,   430,   434,
     438,   442,   446,   450,   454,   458,   460,   463,   465,   467,
     469,   471,   473,   475,   477,   479,   481,   483,   485,   487,
     489,   491,   493,   495,   497,   499,   501,   503,   505,   507,
     509,   511,   513,   515,   517,   519,   521,   523,   525,   527,
     529,   531,   533,   535,   537,   539,   541,   543,   545,   547,
     549,   551,   553,   555,   557,   559,   561,   563,   565,   567,
     569,   571,   573,   575,   577,   579,   581,   583,   585,   587,
     589,   591,   593,   595,   597,   599,   601,   603,   605,   607,
     609,   611,   613,   615,   617,   619,   621,   623,   624,   626,
     630,   632,   635,   638,   640,   642,   644,   646,   649,   651,
     655,   658,   662,   665,   669,   673,   678,   683,   687,   692,
     697,   703,   711,   721,   725,   726,   733,   736,   738,   742,
     747,   752,   757,   760,   764,   769,   774,   779,   785,   791,
     796,   798,   800,   802,   804,   807,   810,   814
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
     167,     0,    -1,    -1,   167,   194,    -1,   167,   168,    -1,
     167,   174,    -1,    -1,   174,   169,   190,    -1,    -1,    -1,
     174,   170,   173,   171,   190,    -1,    22,    41,    69,    41,
      69,    41,    -1,    23,    41,    -1,    20,    41,    -1,   172,
      -1,   173,   172,    -1,    -1,    -1,   181,   141,   175,   184,
     142,   176,   178,    -1,    -1,   181,   177,   178,    -1,   181,
      -1,    -1,    -1,    40,   179,   141,   180,   182,   142,    -1,
      40,    -1,    13,    -1,    38,    13,    -1,    39,    13,    -1,
      16,    -1,    38,    16,    -1,    39,    16,    -1,    14,    16,
      -1,    39,    16,    -1,    -1,   184,    -1,    -1,   182,    69,
     183,   184,    -1,    -1,    24,   185,   196,   187,   198,    -1,
      -1,    26,   186,   196,   198,    -1,    -1,    12,   188,   189,
      -1,    12,   189,    -1,    17,    -1,    18,    -1,    29,    -1,
       5,    41,    -1,   105,   191,   104,    -1,   194,    -1,   209,
      -1,   191,   194,    -1,   191,   209,    -1,    -1,   191,   192,
     190,    -1,    -1,   193,   190,    -1,   195,   101,    -1,    36,
      43,    -1,    36,    43,    99,    -1,    37,    41,    -1,    33,
      40,    69,    40,    -1,    33,    40,    69,    40,    69,    40,
      -1,    33,    40,    -1,    15,    41,     3,    -1,    15,    41,
       3,    69,    41,    69,    41,    -1,    19,    41,    41,    41,
      -1,    25,     3,   101,    -1,   174,   101,    -1,   196,   197,
      -1,   196,   198,   106,   207,    -1,   196,   198,   106,   230,
      -1,    11,    40,    69,    40,    69,    41,    -1,   199,    -1,
     198,    -1,   197,    69,   198,    -1,    40,    -1,    40,   139,
      41,   140,    -1,    40,    94,    95,    -1,    40,    94,    41,
      95,    -1,   200,    -1,   199,   200,    -1,   202,    -1,   204,
      -1,   201,    -1,    14,    -1,    39,    -1,     5,    41,    -1,
      26,    -1,    30,    -1,   203,    -1,    10,    -1,    17,    -1,
      18,    -1,    24,    -1,    29,    -1,    32,    -1,    34,    -1,
     206,    -1,   205,   206,    -1,    66,    -1,    67,    -1,    68,
      -1,    44,    -1,    45,    -1,    46,    -1,    47,    -1,    48,
      -1,    49,    -1,    50,    -1,    51,    -1,    52,    -1,    53,
      -1,    54,    -1,    55,    -1,    56,    -1,    57,    -1,    58,
      -1,    59,    -1,    60,    -1,    61,    -1,    62,    -1,    63,
      -1,    64,    -1,    65,    -1,   105,   208,   104,    -1,   105,
     207,   104,    -1,   230,    -1,   208,    69,   230,    -1,   210,
     101,    -1,    40,   100,    -1,   214,   210,   101,    -1,    -1,
     212,   141,   223,   142,   211,    69,   223,    69,   141,   222,
     142,    -1,   212,   223,    69,   141,   222,   142,    -1,   212,
     223,    69,   141,   142,    -1,   212,   222,    -1,   212,    -1,
      -1,     4,   213,   215,    -1,     4,    -1,    70,    40,    -1,
      70,   102,    40,    -1,    70,    40,    75,    -1,    70,    40,
      73,    -1,    70,    40,    76,    -1,    70,    40,    74,    -1,
      70,    40,    78,    -1,    70,    40,    83,    -1,    70,    40,
      87,    -1,    70,    40,    84,    -1,    70,    40,    91,    -1,
      70,    40,    92,    -1,    70,    40,    93,    -1,   216,    -1,
     216,   215,    -1,   204,    -1,   221,    -1,   203,    -1,   218,
      -1,   125,    -1,   127,    -1,   126,    -1,   118,    -1,    96,
      -1,   145,    -1,   146,    -1,   147,    -1,   148,    -1,   149,
      -1,   150,    -1,   119,    -1,   120,    -1,   121,    -1,   122,
      -1,   123,    -1,   124,    -1,   143,    -1,   144,    -1,   151,
      -1,   152,    -1,   217,    -1,   153,    -1,    71,    -1,    72,
      -1,   154,    -1,   155,    -1,   156,    -1,   157,    -1,   158,
      -1,   159,    -1,   160,    -1,   161,    -1,   162,    -1,   163,
      -1,   164,    -1,   165,    -1,   129,    -1,   128,    -1,   130,
      -1,   131,    -1,   132,    -1,   133,    -1,   134,    -1,   135,
      -1,   136,    -1,   137,    -1,   138,    -1,   219,    -1,   220,
      -1,   110,    -1,   111,    -1,   112,    -1,   113,    -1,   114,
      -1,   115,    -1,   116,    -1,   117,    -1,    73,    -1,    74,
      -1,    75,    -1,    76,    -1,    77,    -1,    78,    -1,    79,
      -1,    80,    -1,    81,    -1,    82,    -1,    83,    -1,    84,
      -1,    85,    -1,    86,    -1,    87,    -1,    88,    -1,    89,
      -1,    90,    -1,    -1,   223,    -1,   223,    69,   222,    -1,
      40,    -1,   102,    40,    -1,    98,    40,    -1,   228,    -1,
     230,    -1,   227,    -1,   224,    -1,    98,   224,    -1,   225,
      -1,    40,    99,    41,    -1,    40,    79,    -1,    98,    40,
      79,    -1,    40,    81,    -1,    98,    40,    81,    -1,    40,
     103,    40,    -1,    40,   103,    40,    79,    -1,    40,   103,
      40,    81,    -1,    40,   108,    40,    -1,    40,   108,    40,
      79,    -1,    40,   108,    40,    81,    -1,   105,    40,    69,
      40,   104,    -1,   105,    40,    69,    40,    69,    40,   104,
      -1,   105,    40,    69,    40,    69,    40,    69,    40,   104,
      -1,   105,    40,   104,    -1,    -1,    94,    40,    69,   226,
     224,    95,    -1,    97,   109,    -1,    97,    -1,    94,   231,
      95,    -1,    40,    94,   231,    95,    -1,    40,    94,   230,
      95,    -1,    40,    94,   229,    95,    -1,    98,   228,    -1,
      40,    99,    40,    -1,    40,    99,    40,    79,    -1,    40,
      99,    40,    81,    -1,    40,    99,   106,    40,    -1,    40,
      99,   106,    40,    79,    -1,    40,    99,   106,    40,    81,
      -1,    40,    99,   106,    41,    -1,    41,    -1,    42,    -1,
      43,    -1,    40,    -1,    40,    79,    -1,    40,    81,    -1,
      40,    99,    41,    -1,    41,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   218,   218,   219,   220,   221,   224,   224,   225,   225,
     225,   228,   231,   232,   235,   236,   239,   239,   239,   240,
     240,   241,   244,   244,   244,   245,   248,   249,   250,   251,
     252,   253,   254,   255,   258,   259,   260,   260,   262,   262,
     263,   263,   265,   266,   267,   269,   270,   271,   273,   275,
     277,   278,   279,   280,   281,   281,   282,   282,   285,   286,
     287,   288,   289,   290,   291,   292,   293,   294,   295,   296,
     299,   300,   301,   302,   305,   307,   308,   310,   311,   323,
     324,   327,   328,   330,   331,   332,   333,   334,   337,   339,
     340,   341,   344,   345,   346,   347,   348,   349,   350,   353,
     354,   357,   358,   359,   362,   363,   364,   365,   366,   367,
     368,   369,   370,   371,   372,   373,   374,   375,   376,   377,
     378,   379,   380,   381,   382,   383,   386,   387,   389,   390,
     392,   393,   394,   396,   396,   397,   398,   399,   400,   403,
     403,   404,   406,   407,   408,   409,   410,   411,   412,   413,
     414,   415,   416,   417,   418,   421,   422,   424,   425,   426,
     427,   428,   429,   430,   431,   432,   433,   434,   435,   436,
     437,   438,   439,   440,   441,   442,   443,   444,   445,   446,
     447,   448,   449,   450,   451,   452,   453,   454,   455,   456,
     457,   458,   459,   460,   461,   462,   463,   464,   467,   468,
     469,   470,   471,   472,   473,   474,   475,   476,   477,   480,
     481,   483,   484,   485,   486,   489,   490,   491,   492,   495,
     496,   497,   498,   499,   500,   501,   502,   503,   504,   505,
     506,   507,   508,   509,   510,   511,   512,   515,   516,   517,
     519,   520,   521,   522,   523,   524,   525,   526,   527,   528,
     529,   530,   531,   532,   533,   534,   535,   536,   537,   538,
     541,   542,   543,   544,   547,   547,   552,   553,   556,   557,
     558,   559,   560,   563,   564,   565,   566,   567,   568,   569,
     572,   573,   574,   577,   578,   579,   580,   581
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "STRING", "OPCODE", "ALIGN_DIRECTIVE",
  "BRANCHTARGETS_DIRECTIVE", "BYTE_DIRECTIVE", "CALLPROTOTYPE_DIRECTIVE",
  "CALLTARGETS_DIRECTIVE", "CONST_DIRECTIVE", "CONSTPTR_DIRECTIVE",
  "PTR_DIRECTIVE", "ENTRY_DIRECTIVE", "EXTERN_DIRECTIVE", "FILE_DIRECTIVE",
  "FUNC_DIRECTIVE", "GLOBAL_DIRECTIVE", "LOCAL_DIRECTIVE", "LOC_DIRECTIVE",
  "MAXNCTAPERSM_DIRECTIVE", "MAXNNREG_DIRECTIVE", "MAXNTID_DIRECTIVE",
  "MINNCTAPERSM_DIRECTIVE", "PARAM_DIRECTIVE", "PRAGMA_DIRECTIVE",
  "REG_DIRECTIVE", "REQNTID_DIRECTIVE", "SECTION_DIRECTIVE",
  "SHARED_DIRECTIVE", "SREG_DIRECTIVE", "STRUCT_DIRECTIVE",
  "SURF_DIRECTIVE", "TARGET_DIRECTIVE", "TEX_DIRECTIVE", "UNION_DIRECTIVE",
  "VERSION_DIRECTIVE", "ADDRESS_SIZE_DIRECTIVE", "VISIBLE_DIRECTIVE",
  "WEAK_DIRECTIVE", "IDENTIFIER", "INT_OPERAND", "FLOAT_OPERAND",
  "DOUBLE_OPERAND", "S8_TYPE", "S16_TYPE", "S32_TYPE", "S64_TYPE",
  "U8_TYPE", "U16_TYPE", "U32_TYPE", "U64_TYPE", "F16_TYPE", "F32_TYPE",
  "F64_TYPE", "FF64_TYPE", "B8_TYPE", "B16_TYPE", "B32_TYPE", "B64_TYPE",
  "BB64_TYPE", "BB128_TYPE", "PRED_TYPE", "TEXREF_TYPE", "SAMPLERREF_TYPE",
  "SURFREF_TYPE", "V2_TYPE", "V3_TYPE", "V4_TYPE", "COMMA", "PRED",
  "HALF_OPTION", "EXTP_OPTION", "EQ_OPTION", "NE_OPTION", "LT_OPTION",
  "LE_OPTION", "GT_OPTION", "GE_OPTION", "LO_OPTION", "LS_OPTION",
  "HI_OPTION", "HS_OPTION", "EQU_OPTION", "NEU_OPTION", "LTU_OPTION",
  "LEU_OPTION", "GTU_OPTION", "GEU_OPTION", "NUM_OPTION", "NAN_OPTION",
  "CF_OPTION", "SF_OPTION", "NSF_OPTION", "LEFT_SQUARE_BRACKET",
  "RIGHT_SQUARE_BRACKET", "WIDE_OPTION", "SPECIAL_REGISTER", "MINUS",
  "PLUS", "COLON", "SEMI_COLON", "EXCLAMATION", "PIPE", "RIGHT_BRACE",
  "LEFT_BRACE", "EQUALS", "PERIOD", "BACKSLASH", "DIMENSION_MODIFIER",
  "RN_OPTION", "RZ_OPTION", "RM_OPTION", "RP_OPTION", "RNI_OPTION",
  "RZI_OPTION", "RMI_OPTION", "RPI_OPTION", "UNI_OPTION",
  "GEOM_MODIFIER_1D", "GEOM_MODIFIER_2D", "GEOM_MODIFIER_3D", "SAT_OPTION",
  "FTZ_OPTION", "NEG_OPTION", "SYNC_OPTION", "RED_OPTION", "ARRIVE_OPTION",
  "ATOMIC_POPC", "ATOMIC_AND", "ATOMIC_OR", "ATOMIC_XOR", "ATOMIC_CAS",
  "ATOMIC_EXCH", "ATOMIC_ADD", "ATOMIC_INC", "ATOMIC_DEC", "ATOMIC_MIN",
  "ATOMIC_MAX", "LEFT_ANGLE_BRACKET", "RIGHT_ANGLE_BRACKET", "LEFT_PAREN",
  "RIGHT_PAREN", "APPROX_OPTION", "FULL_OPTION", "ANY_OPTION",
  "ALL_OPTION", "BALLOT_OPTION", "GLOBAL_OPTION", "CTA_OPTION",
  "SYS_OPTION", "EXIT_OPTION", "ABS_OPTION", "TO_OPTION", "CA_OPTION",
  "CG_OPTION", "CS_OPTION", "LU_OPTION", "CV_OPTION", "WB_OPTION",
  "WT_OPTION", "NC_OPTION", "UP_OPTION", "DOWN_OPTION", "BFLY_OPTION",
  "IDX_OPTION", "$accept", "input", "function_defn", "$@1", "$@2", "$@3",
  "block_spec", "block_spec_list", "function_decl", "$@4", "$@5", "$@6",
  "function_ident_param", "$@7", "$@8", "function_decl_header",
  "param_list", "$@9", "param_entry", "$@10", "$@11", "ptr_spec",
  "ptr_space_spec", "ptr_align_spec", "statement_block", "statement_list",
  "$@12", "$@13", "directive_statement", "variable_declaration",
  "variable_spec", "identifier_list", "identifier_spec", "var_spec_list",
  "var_spec", "align_spec", "space_spec", "addressable_spec", "type_spec",
  "vector_spec", "scalar_type", "initializer_list", "literal_list",
  "instruction_statement", "instruction", "$@14", "opcode_spec", "$@15",
  "pred_spec", "option_list", "option", "atomic_operation_spec",
  "rounding_mode", "floating_point_rounding_mode", "integer_rounding_mode",
  "compare_spec", "operand_list", "operand", "vector_operand",
  "tex_operand", "$@16", "builtin_operand", "memory_operand",
  "twin_operand", "literal_operand", "address_expression", 0
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[YYLEX-NUM] -- Internal token number corresponding to
   token YYLEX-NUM.  */
static const yytype_uint16 yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   294,
     295,   296,   297,   298,   299,   300,   301,   302,   303,   304,
     305,   306,   307,   308,   309,   310,   311,   312,   313,   314,
     315,   316,   317,   318,   319,   320,   321,   322,   323,   324,
     325,   326,   327,   328,   329,   330,   331,   332,   333,   334,
     335,   336,   337,   338,   339,   340,   341,   342,   343,   344,
     345,   346,   347,   348,   349,   350,   351,   352,   353,   354,
     355,   356,   357,   358,   359,   360,   361,   362,   363,   364,
     365,   366,   367,   368,   369,   370,   371,   372,   373,   374,
     375,   376,   377,   378,   379,   380,   381,   382,   383,   384,
     385,   386,   387,   388,   389,   390,   391,   392,   393,   394,
     395,   396,   397,   398,   399,   400,   401,   402,   403,   404,
     405,   406,   407,   408,   409,   410,   411,   412,   413,   414,
     415,   416,   417,   418,   419,   420
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,   166,   167,   167,   167,   167,   169,   168,   170,   171,
     168,   172,   172,   172,   173,   173,   175,   176,   174,   177,
     174,   174,   179,   180,   178,   178,   181,   181,   181,   181,
     181,   181,   181,   181,   182,   182,   183,   182,   185,   184,
     186,   184,   187,   187,   187,   188,   188,   188,   189,   190,
     191,   191,   191,   191,   192,   191,   193,   191,   194,   194,
     194,   194,   194,   194,   194,   194,   194,   194,   194,   194,
     195,   195,   195,   195,   196,   197,   197,   198,   198,   198,
     198,   199,   199,   200,   200,   200,   200,   200,   201,   202,
     202,   202,   203,   203,   203,   203,   203,   203,   203,   204,
     204,   205,   205,   205,   206,   206,   206,   206,   206,   206,
     206,   206,   206,   206,   206,   206,   206,   206,   206,   206,
     206,   206,   206,   206,   206,   206,   207,   207,   208,   208,
     209,   209,   209,   211,   210,   210,   210,   210,   210,   213,
     212,   212,   214,   214,   214,   214,   214,   214,   214,   214,
     214,   214,   214,   214,   214,   215,   215,   216,   216,   216,
     216,   216,   216,   216,   216,   216,   216,   216,   216,   216,
     216,   216,   216,   216,   216,   216,   216,   216,   216,   216,
     216,   216,   216,   216,   216,   216,   216,   216,   216,   216,
     216,   216,   216,   216,   216,   216,   216,   216,   217,   217,
     217,   217,   217,   217,   217,   217,   217,   217,   217,   218,
     218,   219,   219,   219,   219,   220,   220,   220,   220,   221,
     221,   221,   221,   221,   221,   221,   221,   221,   221,   221,
     221,   221,   221,   221,   221,   221,   221,   222,   222,   222,
     223,   223,   223,   223,   223,   223,   223,   223,   223,   223,
     223,   223,   223,   223,   223,   223,   223,   223,   223,   223,
     224,   224,   224,   224,   226,   225,   227,   227,   228,   228,
     228,   228,   228,   229,   229,   229,   229,   229,   229,   229,
     230,   230,   230,   231,   231,   231,   231,   231
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     0,     2,     2,     2,     0,     3,     0,     0,
       5,     6,     2,     2,     1,     2,     0,     0,     7,     0,
       3,     1,     0,     0,     6,     1,     1,     2,     2,     1,
       2,     2,     2,     2,     0,     1,     0,     4,     0,     5,
       0,     4,     0,     3,     2,     1,     1,     1,     2,     3,
       1,     1,     2,     2,     0,     3,     0,     2,     2,     2,
       3,     2,     4,     6,     2,     3,     7,     4,     3,     2,
       2,     4,     4,     6,     1,     1,     3,     1,     4,     3,
       4,     1,     2,     1,     1,     1,     1,     1,     2,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       2,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     3,     3,     1,     3,
       2,     2,     3,     0,    11,     6,     5,     2,     1,     0,
       3,     1,     2,     3,     3,     3,     3,     3,     3,     3,
       3,     3,     3,     3,     3,     1,     2,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     0,     1,     3,
       1,     2,     2,     1,     1,     1,     1,     2,     1,     3,
       2,     3,     2,     3,     3,     4,     4,     3,     4,     4,
       5,     7,     9,     3,     0,     6,     2,     1,     3,     4,
       4,     4,     2,     3,     4,     4,     4,     5,     5,     4,
       1,     1,     1,     1,     2,     2,     3,     1
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       2,     0,     1,     0,    92,     0,    26,    86,     0,    29,
      93,    94,     0,    95,     0,    89,    96,    90,    97,     0,
      98,     0,     0,     0,    87,   104,   105,   106,   107,   108,
     109,   110,   111,   112,   113,   114,   115,   116,   117,   118,
     119,   120,   121,   122,   123,   124,   125,   101,   102,   103,
       4,     5,    21,     3,     0,     0,    74,    81,    85,    83,
      91,    84,     0,    99,    88,     0,    32,     0,     0,     0,
      64,    59,    61,    27,    30,    28,    31,    69,     0,     0,
      16,     0,    58,    77,    70,    75,    86,    87,    82,   100,
       0,    65,     0,    68,     0,    60,    56,     7,     0,     0,
       0,    14,     9,     0,    25,    20,     0,     0,     0,     0,
       0,     0,    67,    62,   139,     0,     0,     0,    54,     0,
      50,    51,     0,   138,     0,    13,     0,    12,     0,    15,
      38,    40,     0,     0,     0,    79,     0,    76,   280,   281,
     282,     0,    71,    72,     0,     0,     0,     0,   131,   142,
       0,    49,     0,    52,    53,    57,   130,   240,     0,   267,
       0,     0,     0,     0,   137,   238,   246,   248,   245,   243,
     244,     0,     0,    10,     0,     0,    17,    23,    80,    78,
       0,     0,   128,    73,     0,    63,   184,   185,   219,   220,
     221,   222,   223,   224,   225,   226,   227,   228,   229,   230,
     231,   232,   233,   234,   235,   236,   165,   211,   212,   213,
     214,   215,   216,   217,   218,   164,   172,   173,   174,   175,
     176,   177,   161,   163,   162,   199,   198,   200,   201,   202,
     203,   204,   205,   206,   207,   208,   178,   179,   166,   167,
     168,   169,   170,   171,   180,   181,   183,   186,   187,   188,
     189,   190,   191,   192,   193,   194,   195,   196,   197,   159,
     157,   140,   155,   182,   160,   209,   210,   158,   145,   147,
     144,   146,   148,   149,   151,   150,   152,   153,   154,   143,
      55,   250,   252,     0,     0,     0,     0,   283,   287,     0,
     266,   242,     0,     0,   247,   272,   241,     0,     0,   237,
     132,     0,    42,     0,     0,    34,   127,     0,   126,    66,
     156,   283,   280,     0,     0,     0,   249,   254,   257,   264,
     284,   285,     0,   268,   251,   253,   283,     0,     0,   263,
     133,     0,   239,   238,     0,     0,     0,    41,    18,     0,
      35,   129,     0,   271,   270,   269,   255,   256,   258,   259,
       0,   286,     0,     0,   136,     0,   237,    11,     0,    45,
      46,    47,     0,    44,    39,    36,    24,   273,     0,     0,
       0,   260,     0,   135,    48,    43,     0,   274,   275,   276,
     279,   265,     0,     0,    37,   277,   278,     0,   261,     0,
       0,   237,   262,     0,   134
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,    50,    78,    79,   128,   101,   102,   117,   103,
     304,    81,   105,   133,   305,    52,   339,   376,   132,   174,
     175,   336,   362,   363,    97,   118,   152,   119,    53,    54,
      55,    84,    85,    56,    57,    58,    59,    60,    61,    62,
      63,   142,   181,   121,   122,   353,   123,   147,   124,   261,
     262,   263,   264,   265,   266,   267,   332,   333,   166,   167,
     350,   168,   169,   313,   170,   289
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -288
static const yytype_int16 yypact[] =
{
    -288,   345,  -288,   -27,  -288,   -19,  -288,    12,    55,  -288,
    -288,  -288,   140,  -288,   185,  -288,  -288,  -288,  -288,   166,
    -288,   177,   182,    90,   203,  -288,  -288,  -288,  -288,  -288,
    -288,  -288,  -288,  -288,  -288,  -288,  -288,  -288,  -288,  -288,
    -288,  -288,  -288,  -288,  -288,  -288,  -288,  -288,  -288,  -288,
    -288,   -10,   -34,  -288,   139,   197,   476,  -288,  -288,  -288,
    -288,  -288,   501,  -288,  -288,   204,  -288,   251,   231,   174,
     208,   179,  -288,  -288,  -288,  -288,  -288,  -288,   171,    78,
    -288,   242,  -288,    74,   215,   190,  -288,  -288,  -288,  -288,
     257,   229,   262,  -288,   264,  -288,   410,  -288,   266,   270,
     277,  -288,    78,    11,   176,  -288,    -3,   278,   197,   134,
     279,   306,  -288,   280,   130,   252,     0,   250,   276,   171,
    -288,  -288,   253,   144,   349,  -288,   288,  -288,   171,  -288,
    -288,  -288,   223,   225,   272,  -288,   228,  -288,  -288,  -288,
    -288,   134,  -288,  -288,   331,   304,   336,    -2,  -288,   494,
     346,  -288,   171,  -288,  -288,  -288,  -288,   180,   193,   307,
      -1,   347,   348,   390,  -288,   316,  -288,  -288,  -288,  -288,
    -288,   317,   376,  -288,   476,   476,  -288,  -288,  -288,  -288,
     315,    36,  -288,  -288,   381,  -288,  -288,  -288,  -288,  -288,
    -288,  -288,  -288,  -288,  -288,  -288,  -288,  -288,  -288,  -288,
    -288,  -288,  -288,  -288,  -288,  -288,  -288,  -288,  -288,  -288,
    -288,  -288,  -288,  -288,  -288,  -288,  -288,  -288,  -288,  -288,
    -288,  -288,  -288,  -288,  -288,  -288,  -288,  -288,  -288,  -288,
    -288,  -288,  -288,  -288,  -288,  -288,  -288,  -288,  -288,  -288,
    -288,  -288,  -288,  -288,  -288,  -288,  -288,  -288,  -288,  -288,
    -288,  -288,  -288,  -288,  -288,  -288,  -288,  -288,  -288,  -288,
    -288,  -288,    -2,  -288,  -288,  -288,  -288,  -288,  -288,  -288,
    -288,  -288,  -288,  -288,  -288,  -288,  -288,  -288,  -288,  -288,
    -288,  -288,  -288,   -17,   396,   398,   401,   111,  -288,   350,
    -288,   115,   207,    97,  -288,  -288,  -288,    70,   309,   158,
    -288,   383,   441,   197,   242,    11,  -288,   202,  -288,  -288,
    -288,   -70,  -288,   384,   387,   388,  -288,   136,   172,  -288,
    -288,  -288,   444,  -288,  -288,  -288,   126,   395,   451,  -288,
    -288,   124,  -288,   427,   456,     2,   197,  -288,  -288,   -52,
    -288,  -288,    -7,  -288,  -288,  -288,  -288,  -288,  -288,  -288,
     393,  -288,   100,   430,  -288,   359,   390,  -288,   462,  -288,
    -288,  -288,   499,  -288,  -288,  -288,  -288,   183,   217,   412,
     469,  -288,   390,  -288,  -288,  -288,    11,  -288,  -288,   186,
    -288,  -288,   110,   442,  -288,  -288,  -288,   472,  -288,   372,
     413,   390,  -288,   374,  -288
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -288,  -288,  -288,  -288,  -288,  -288,   416,  -288,   513,  -288,
    -288,  -288,   267,  -288,  -288,  -288,  -288,  -288,  -287,  -288,
    -288,  -288,  -288,   157,    84,  -288,  -288,  -288,    93,  -288,
      95,  -288,  -106,  -288,   517,  -288,  -288,   -80,   -79,  -288,
     512,   434,  -288,   458,   455,  -288,  -288,  -288,  -288,   318,
    -288,  -288,  -288,  -288,  -288,  -288,  -123,  -122,  -157,  -288,
    -288,  -288,  -155,  -288,  -105,   299
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -142
static const yytype_int16 yytable[] =
{
     164,   165,   137,   294,   143,   295,   -19,   358,     4,   320,
      -8,   321,    -8,    -8,    64,    10,    11,   365,   340,   359,
     360,    65,    13,   311,   312,   139,   140,    16,    66,   342,
      18,   361,    20,   367,   351,   130,   182,   131,   134,   291,
     149,   298,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    34,    35,    36,    37,    38,    39,    40,    41,    42,
      43,    44,    45,    46,    47,    48,    49,   259,   260,   186,
     187,   188,   189,   190,   191,   192,   193,   194,   195,   196,
     197,   198,   199,   200,   201,   202,   203,   204,   205,   384,
     366,    77,   135,   292,   206,    -6,    67,   293,    98,   368,
      99,   100,   150,    73,   162,   307,    74,    80,   207,   208,
     209,   210,   211,   212,   213,   214,   215,   216,   217,   218,
     219,   220,   221,   222,   223,   224,   225,   226,   227,   228,
     229,   230,   231,   232,   233,   234,   235,   327,   295,   328,
     308,   236,   237,   238,   239,   240,   241,   242,   243,   244,
     245,   246,   247,   248,   249,   250,   251,   252,   253,   254,
     255,   256,   257,   258,   157,   138,   139,   140,   106,   370,
    -141,  -141,  -141,  -141,   329,   138,   139,   140,   314,   387,
     319,    68,   259,   260,   157,   138,   139,   140,    69,   120,
     320,   292,   321,   369,   324,   293,   325,   337,   157,   138,
     139,   140,   341,   155,   371,   320,    70,   321,   355,   283,
     322,   153,   173,   107,   388,   346,    75,   347,   158,    76,
      71,   159,   160,    72,  -141,   322,   161,  -141,  -141,   162,
     364,  -141,  -141,   287,   288,  -141,   280,    83,   158,   141,
      82,   159,   160,   138,   139,   140,   161,   326,   288,   162,
     383,   348,   158,   349,    91,   159,   160,   379,   380,   281,
     161,   282,   377,   162,   378,   385,   354,   386,   393,   302,
     303,  -141,    92,    90,   283,    93,    96,    94,    95,   284,
     114,     3,   104,   285,   108,   163,     4,     5,   286,     6,
       7,     8,     9,    10,    11,    12,   109,   110,   111,   331,
      13,    14,    15,   112,   113,    16,    17,   125,    18,    19,
      20,   126,    21,    22,    23,    24,   115,   -22,   127,   136,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,     2,   116,   145,   144,   146,
       3,    77,   148,   114,   156,     4,     5,   172,     6,     7,
       8,     9,    10,    11,    12,   176,   177,   178,   179,    13,
      14,    15,   183,   184,    16,    17,   185,    18,    19,    20,
     151,    21,    22,    23,    24,   299,   279,   296,   297,    25,
      26,    27,    28,    29,    30,    31,    32,    33,    34,    35,
      36,    37,    38,    39,    40,    41,    42,    43,    44,    45,
      46,    47,    48,    49,   114,     3,   290,   301,   300,   306,
       4,     5,   309,     6,     7,     8,     9,    10,    11,    12,
     157,   138,   139,   140,    13,    14,    15,   316,   317,    16,
      17,   318,    18,    19,    20,   323,    21,    22,    23,    24,
     115,   330,   334,   335,    25,    26,    27,    28,    29,    30,
      31,    32,    33,    34,    35,    36,    37,    38,    39,    40,
      41,    42,    43,    44,    45,    46,    47,    48,    49,   343,
     116,     3,   344,   345,   158,   351,     4,   159,   160,   283,
      86,   352,   161,    10,    11,   162,   356,   357,   162,   372,
      13,   373,    15,   374,   358,    16,    17,   381,    18,   382,
      20,   389,   390,   391,    51,    87,   394,   392,   129,   375,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    25,    26,    27,    28,    29,
      30,    31,    32,    33,    34,    35,    36,    37,    38,    39,
      40,    41,    42,    43,    44,    45,    46,   268,   269,   270,
     271,   338,   272,    88,    89,   180,   154,   273,   274,   171,
     310,   275,   315,     0,     0,   276,   277,   278
};

static const yytype_int16 yycheck[] =
{
     123,   123,   108,   160,   109,   160,    40,     5,    10,    79,
      20,    81,    22,    23,    41,    17,    18,    69,   305,    17,
      18,    40,    24,    40,    41,    42,    43,    29,    16,    99,
      32,    29,    34,    40,    41,    24,   141,    26,    41,    40,
      40,   163,    44,    45,    46,    47,    48,    49,    50,    51,
      52,    53,    54,    55,    56,    57,    58,    59,    60,    61,
      62,    63,    64,    65,    66,    67,    68,   147,   147,    71,
      72,    73,    74,    75,    76,    77,    78,    79,    80,    81,
      82,    83,    84,    85,    86,    87,    88,    89,    90,   376,
     142,   101,    95,    94,    96,   105,    41,    98,    20,   106,
      22,    23,   102,    13,   105,    69,    16,   141,   110,   111,
     112,   113,   114,   115,   116,   117,   118,   119,   120,   121,
     122,   123,   124,   125,   126,   127,   128,   129,   130,   131,
     132,   133,   134,   135,   136,   137,   138,    40,   293,    69,
     104,   143,   144,   145,   146,   147,   148,   149,   150,   151,
     152,   153,   154,   155,   156,   157,   158,   159,   160,   161,
     162,   163,   164,   165,    40,    41,    42,    43,    94,    69,
      40,    41,    42,    43,   104,    41,    42,    43,   283,    69,
      69,    41,   262,   262,    40,    41,    42,    43,     3,    96,
      79,    94,    81,   350,    79,    98,    81,   303,    40,    41,
      42,    43,   307,   119,   104,    79,    40,    81,   331,    94,
      99,   118,   128,   139,   104,    79,    13,    81,    94,    16,
      43,    97,    98,    41,    94,    99,   102,    97,    98,   105,
     336,   101,   102,    40,    41,   105,   152,    40,    94,   105,
     101,    97,    98,    41,    42,    43,   102,    40,    41,   105,
     372,    79,    94,    81,     3,    97,    98,    40,    41,    79,
     102,    81,    79,   105,    81,    79,   142,    81,   391,   174,
     175,   141,    41,    69,    94,   101,   105,    69,    99,    99,
       4,     5,    40,   103,    69,   141,    10,    11,   108,    13,
      14,    15,    16,    17,    18,    19,   106,    40,    69,   141,
      24,    25,    26,    41,    40,    29,    30,    41,    32,    33,
      34,    41,    36,    37,    38,    39,    40,   141,    41,    41,
      44,    45,    46,    47,    48,    49,    50,    51,    52,    53,
      54,    55,    56,    57,    58,    59,    60,    61,    62,    63,
      64,    65,    66,    67,    68,     0,    70,    41,    69,    69,
       5,   101,   100,     4,   101,    10,    11,    69,    13,    14,
      15,    16,    17,    18,    19,   142,   141,    95,   140,    24,
      25,    26,    41,    69,    29,    30,    40,    32,    33,    34,
     104,    36,    37,    38,    39,    69,    40,    40,    40,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,     4,     5,   109,    41,   101,   104,
      10,    11,    41,    13,    14,    15,    16,    17,    18,    19,
      40,    41,    42,    43,    24,    25,    26,    41,    40,    29,
      30,    40,    32,    33,    34,    95,    36,    37,    38,    39,
      40,   142,    69,    12,    44,    45,    46,    47,    48,    49,
      50,    51,    52,    53,    54,    55,    56,    57,    58,    59,
      60,    61,    62,    63,    64,    65,    66,    67,    68,    95,
      70,     5,    95,    95,    94,    41,    10,    97,    98,    94,
      14,    40,   102,    17,    18,   105,    69,    41,   105,    69,
      24,   142,    26,    41,     5,    29,    30,    95,    32,    40,
      34,    69,    40,   141,     1,    39,   142,   104,   102,   362,
      44,    45,    46,    47,    48,    49,    50,    51,    52,    53,
      54,    55,    56,    57,    58,    59,    60,    61,    62,    63,
      64,    65,    66,    67,    68,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,    55,    56,    57,    58,
      59,    60,    61,    62,    63,    64,    65,    73,    74,    75,
      76,   304,    78,    56,    62,   141,   118,    83,    84,   124,
     262,    87,   283,    -1,    -1,    91,    92,    93
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,   167,     0,     5,    10,    11,    13,    14,    15,    16,
      17,    18,    19,    24,    25,    26,    29,    30,    32,    33,
      34,    36,    37,    38,    39,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,    55,    56,    57,    58,
      59,    60,    61,    62,    63,    64,    65,    66,    67,    68,
     168,   174,   181,   194,   195,   196,   199,   200,   201,   202,
     203,   204,   205,   206,    41,    40,    16,    41,    41,     3,
      40,    43,    41,    13,    16,    13,    16,   101,   169,   170,
     141,   177,   101,    40,   197,   198,    14,    39,   200,   206,
      69,     3,    41,   101,    69,    99,   105,   190,    20,    22,
      23,   172,   173,   175,    40,   178,    94,   139,    69,   106,
      40,    69,    41,    40,     4,    40,    70,   174,   191,   193,
     194,   209,   210,   212,   214,    41,    41,    41,   171,   172,
      24,    26,   184,   179,    41,    95,    41,   198,    41,    42,
      43,   105,   207,   230,    69,    41,    69,   213,   100,    40,
     102,   104,   192,   194,   209,   190,   101,    40,    94,    97,
      98,   102,   105,   141,   222,   223,   224,   225,   227,   228,
     230,   210,    69,   190,   185,   186,   142,   141,    95,   140,
     207,   208,   230,    41,    69,    40,    71,    72,    73,    74,
      75,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    96,   110,   111,   112,
     113,   114,   115,   116,   117,   118,   119,   120,   121,   122,
     123,   124,   125,   126,   127,   128,   129,   130,   131,   132,
     133,   134,   135,   136,   137,   138,   143,   144,   145,   146,
     147,   148,   149,   150,   151,   152,   153,   154,   155,   156,
     157,   158,   159,   160,   161,   162,   163,   164,   165,   203,
     204,   215,   216,   217,   218,   219,   220,   221,    73,    74,
      75,    76,    78,    83,    84,    87,    91,    92,    93,    40,
     190,    79,    81,    94,    99,   103,   108,    40,    41,   231,
     109,    40,    94,    98,   224,   228,    40,    40,   223,    69,
     101,    41,   196,   196,   176,   180,   104,    69,   104,    41,
     215,    40,    41,   229,   230,   231,    41,    40,    40,    69,
      79,    81,    99,    95,    79,    81,    40,    40,    69,   104,
     142,   141,   222,   223,    69,    12,   187,   198,   178,   182,
     184,   230,    99,    95,    95,    95,    79,    81,    79,    81,
     226,    41,    40,   211,   142,   222,    69,    41,     5,    17,
      18,    29,   188,   189,   198,    69,   142,    40,   106,   224,
      69,   104,    69,   142,    41,   189,   183,    79,    81,    40,
      41,    95,    40,   223,   184,    79,    81,    69,   104,    69,
      40,   141,   104,   222,   142
};

#define yyerrok		(yyerrstatus = 0)
#define yyclearin	(yychar = YYEMPTY)
#define YYEMPTY		(-2)
#define YYEOF		0

#define YYACCEPT	goto yyacceptlab
#define YYABORT		goto yyabortlab
#define YYERROR		goto yyerrorlab


/* Like YYERROR except do call yyerror.  This remains here temporarily
   to ease the transition to the new meaning of YYERROR, for GCC.
   Once GCC version 2 has supplanted version 1, this can go.  */

#define YYFAIL		goto yyerrlab

#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)					\
do								\
  if (yychar == YYEMPTY && yylen == 1)				\
    {								\
      yychar = (Token);						\
      yylval = (Value);						\
      yytoken = YYTRANSLATE (yychar);				\
      YYPOPSTACK (1);						\
      goto yybackup;						\
    }								\
  else								\
    {								\
      yyerror (YY_("syntax error: cannot back up")); \
      YYERROR;							\
    }								\
while (YYID (0))


#define YYTERROR	1
#define YYERRCODE	256


/* YYLLOC_DEFAULT -- Set CURRENT to span from RHS[1] to RHS[N].
   If N is 0, then set CURRENT to the empty location which ends
   the previous symbol: RHS[0] (always defined).  */

#define YYRHSLOC(Rhs, K) ((Rhs)[K])
#ifndef YYLLOC_DEFAULT
# define YYLLOC_DEFAULT(Current, Rhs, N)				\
    do									\
      if (YYID (N))                                                    \
	{								\
	  (Current).first_line   = YYRHSLOC (Rhs, 1).first_line;	\
	  (Current).first_column = YYRHSLOC (Rhs, 1).first_column;	\
	  (Current).last_line    = YYRHSLOC (Rhs, N).last_line;		\
	  (Current).last_column  = YYRHSLOC (Rhs, N).last_column;	\
	}								\
      else								\
	{								\
	  (Current).first_line   = (Current).last_line   =		\
	    YYRHSLOC (Rhs, 0).last_line;				\
	  (Current).first_column = (Current).last_column =		\
	    YYRHSLOC (Rhs, 0).last_column;				\
	}								\
    while (YYID (0))
#endif


/* YY_LOCATION_PRINT -- Print the location on the stream.
   This macro was not mandated originally: define only if we know
   we won't break user code: when these are the locations we know.  */

#ifndef YY_LOCATION_PRINT
# if YYLTYPE_IS_TRIVIAL
#  define YY_LOCATION_PRINT(File, Loc)			\
     fprintf (File, "%d.%d-%d.%d",			\
	      (Loc).first_line, (Loc).first_column,	\
	      (Loc).last_line,  (Loc).last_column)
# else
#  define YY_LOCATION_PRINT(File, Loc) ((void) 0)
# endif
#endif


/* YYLEX -- calling `yylex' with the right arguments.  */

#ifdef YYLEX_PARAM
# define YYLEX yylex (YYLEX_PARAM)
#else
# define YYLEX yylex ()
#endif

/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)			\
do {						\
  if (yydebug)					\
    YYFPRINTF Args;				\
} while (YYID (0))

# define YY_SYMBOL_PRINT(Title, Type, Value, Location)			  \
do {									  \
  if (yydebug)								  \
    {									  \
      YYFPRINTF (stderr, "%s ", Title);					  \
      yy_symbol_print (stderr,						  \
		  Type, Value); \
      YYFPRINTF (stderr, "\n");						  \
    }									  \
} while (YYID (0))


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_value_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_value_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (!yyvaluep)
    return;
# ifdef YYPRINT
  if (yytype < YYNTOKENS)
    YYPRINT (yyoutput, yytoknum[yytype], *yyvaluep);
# else
  YYUSE (yyoutput);
# endif
  switch (yytype)
    {
      default:
	break;
    }
}


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (yytype < YYNTOKENS)
    YYFPRINTF (yyoutput, "token %s (", yytname[yytype]);
  else
    YYFPRINTF (yyoutput, "nterm %s (", yytname[yytype]);

  yy_symbol_value_print (yyoutput, yytype, yyvaluep);
  YYFPRINTF (yyoutput, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_stack_print (yytype_int16 *yybottom, yytype_int16 *yytop)
#else
static void
yy_stack_print (yybottom, yytop)
    yytype_int16 *yybottom;
    yytype_int16 *yytop;
#endif
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)				\
do {								\
  if (yydebug)							\
    yy_stack_print ((Bottom), (Top));				\
} while (YYID (0))


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_reduce_print (YYSTYPE *yyvsp, int yyrule)
#else
static void
yy_reduce_print (yyvsp, yyrule)
    YYSTYPE *yyvsp;
    int yyrule;
#endif
{
  int yynrhs = yyr2[yyrule];
  int yyi;
  unsigned long int yylno = yyrline[yyrule];
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %lu):\n",
	     yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr, yyrhs[yyprhs[yyrule] + yyi],
		       &(yyvsp[(yyi + 1) - (yynrhs)])
		       		       );
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)		\
do {					\
  if (yydebug)				\
    yy_reduce_print (yyvsp, Rule); \
} while (YYID (0))

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args)
# define YY_SYMBOL_PRINT(Title, Type, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef	YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif



#if YYERROR_VERBOSE

# ifndef yystrlen
#  if defined __GLIBC__ && defined _STRING_H
#   define yystrlen strlen
#  else
/* Return the length of YYSTR.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static YYSIZE_T
yystrlen (const char *yystr)
#else
static YYSIZE_T
yystrlen (yystr)
    const char *yystr;
#endif
{
  YYSIZE_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++)
    continue;
  return yylen;
}
#  endif
# endif

# ifndef yystpcpy
#  if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#   define yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static char *
yystpcpy (char *yydest, const char *yysrc)
#else
static char *
yystpcpy (yydest, yysrc)
    char *yydest;
    const char *yysrc;
#endif
{
  char *yyd = yydest;
  const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#  endif
# endif

# ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYSIZE_T
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      YYSIZE_T yyn = 0;
      char const *yyp = yystr;

      for (;;)
	switch (*++yyp)
	  {
	  case '\'':
	  case ',':
	    goto do_not_strip_quotes;

	  case '\\':
	    if (*++yyp != '\\')
	      goto do_not_strip_quotes;
	    /* Fall through.  */
	  default:
	    if (yyres)
	      yyres[yyn] = *yyp;
	    yyn++;
	    break;

	  case '"':
	    if (yyres)
	      yyres[yyn] = '\0';
	    return yyn;
	  }
    do_not_strip_quotes: ;
    }

  if (! yyres)
    return yystrlen (yystr);

  return yystpcpy (yyres, yystr) - yyres;
}
# endif

/* Copy into YYRESULT an error message about the unexpected token
   YYCHAR while in state YYSTATE.  Return the number of bytes copied,
   including the terminating null byte.  If YYRESULT is null, do not
   copy anything; just return the number of bytes that would be
   copied.  As a special case, return 0 if an ordinary "syntax error"
   message will do.  Return YYSIZE_MAXIMUM if overflow occurs during
   size calculation.  */
static YYSIZE_T
yysyntax_error (char *yyresult, int yystate, int yychar)
{
  int yyn = yypact[yystate];

  if (! (YYPACT_NINF < yyn && yyn <= YYLAST))
    return 0;
  else
    {
      int yytype = YYTRANSLATE (yychar);
      YYSIZE_T yysize0 = yytnamerr (0, yytname[yytype]);
      YYSIZE_T yysize = yysize0;
      YYSIZE_T yysize1;
      int yysize_overflow = 0;
      enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
      char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
      int yyx;

# if 0
      /* This is so xgettext sees the translatable formats that are
	 constructed on the fly.  */
      YY_("syntax error, unexpected %s");
      YY_("syntax error, unexpected %s, expecting %s");
      YY_("syntax error, unexpected %s, expecting %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s");
# endif
      char *yyfmt;
      char const *yyf;
      static char const yyunexpected[] = "syntax error, unexpected %s";
      static char const yyexpecting[] = ", expecting %s";
      static char const yyor[] = " or %s";
      char yyformat[sizeof yyunexpected
		    + sizeof yyexpecting - 1
		    + ((YYERROR_VERBOSE_ARGS_MAXIMUM - 2)
		       * (sizeof yyor - 1))];
      char const *yyprefix = yyexpecting;

      /* Start YYX at -YYN if negative to avoid negative indexes in
	 YYCHECK.  */
      int yyxbegin = yyn < 0 ? -yyn : 0;

      /* Stay within bounds of both yycheck and yytname.  */
      int yychecklim = YYLAST - yyn + 1;
      int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
      int yycount = 1;

      yyarg[0] = yytname[yytype];
      yyfmt = yystpcpy (yyformat, yyunexpected);

      for (yyx = yyxbegin; yyx < yyxend; ++yyx)
	if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR)
	  {
	    if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
	      {
		yycount = 1;
		yysize = yysize0;
		yyformat[sizeof yyunexpected - 1] = '\0';
		break;
	      }
	    yyarg[yycount++] = yytname[yyx];
	    yysize1 = yysize + yytnamerr (0, yytname[yyx]);
	    yysize_overflow |= (yysize1 < yysize);
	    yysize = yysize1;
	    yyfmt = yystpcpy (yyfmt, yyprefix);
	    yyprefix = yyor;
	  }

      yyf = YY_(yyformat);
      yysize1 = yysize + yystrlen (yyf);
      yysize_overflow |= (yysize1 < yysize);
      yysize = yysize1;

      if (yysize_overflow)
	return YYSIZE_MAXIMUM;

      if (yyresult)
	{
	  /* Avoid sprintf, as that infringes on the user's name space.
	     Don't have undefined behavior even if the translation
	     produced a string with the wrong number of "%s"s.  */
	  char *yyp = yyresult;
	  int yyi = 0;
	  while ((*yyp = *yyf) != '\0')
	    {
	      if (*yyp == '%' && yyf[1] == 's' && yyi < yycount)
		{
		  yyp += yytnamerr (yyp, yyarg[yyi++]);
		  yyf += 2;
		}
	      else
		{
		  yyp++;
		  yyf++;
		}
	    }
	}
      return yysize;
    }
}
#endif /* YYERROR_VERBOSE */


/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yydestruct (const char *yymsg, int yytype, YYSTYPE *yyvaluep)
#else
static void
yydestruct (yymsg, yytype, yyvaluep)
    const char *yymsg;
    int yytype;
    YYSTYPE *yyvaluep;
#endif
{
  YYUSE (yyvaluep);

  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yytype, yyvaluep, yylocationp);

  switch (yytype)
    {

      default:
	break;
    }
}

/* Prevent warnings from -Wmissing-prototypes.  */
#ifdef YYPARSE_PARAM
#if defined __STDC__ || defined __cplusplus
int yyparse (void *YYPARSE_PARAM);
#else
int yyparse ();
#endif
#else /* ! YYPARSE_PARAM */
#if defined __STDC__ || defined __cplusplus
int yyparse (void);
#else
int yyparse ();
#endif
#endif /* ! YYPARSE_PARAM */


/* The lookahead symbol.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval;

/* Number of syntax errors so far.  */
int yynerrs;



/*-------------------------.
| yyparse or yypush_parse.  |
`-------------------------*/

#ifdef YYPARSE_PARAM
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void *YYPARSE_PARAM)
#else
int
yyparse (YYPARSE_PARAM)
    void *YYPARSE_PARAM;
#endif
#else /* ! YYPARSE_PARAM */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void)
#else
int
yyparse ()

#endif
#endif
{


    int yystate;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus;

    /* The stacks and their tools:
       `yyss': related to states.
       `yyvs': related to semantic values.

       Refer to the stacks thru separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* The state stack.  */
    yytype_int16 yyssa[YYINITDEPTH];
    yytype_int16 *yyss;
    yytype_int16 *yyssp;

    /* The semantic value stack.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs;
    YYSTYPE *yyvsp;

    YYSIZE_T yystacksize;

  int yyn;
  int yyresult;
  /* Lookahead token as an internal (translated) token number.  */
  int yytoken;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;

#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYSIZE_T yymsg_alloc = sizeof yymsgbuf;
#endif

#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  yytoken = 0;
  yyss = yyssa;
  yyvs = yyvsa;
  yystacksize = YYINITDEPTH;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY; /* Cause a token to be read.  */

  /* Initialize stack pointers.
     Waste one element of value and location stack
     so that they stay on the same level as the state stack.
     The wasted elements are never initialized.  */
  yyssp = yyss;
  yyvsp = yyvs;

  goto yysetstate;

/*------------------------------------------------------------.
| yynewstate -- Push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
 yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;

 yysetstate:
  *yyssp = yystate;

  if (yyss + yystacksize - 1 <= yyssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T yysize = yyssp - yyss + 1;

#ifdef yyoverflow
      {
	/* Give user a chance to reallocate the stack.  Use copies of
	   these so that the &'s don't force the real ones into
	   memory.  */
	YYSTYPE *yyvs1 = yyvs;
	yytype_int16 *yyss1 = yyss;

	/* Each stack pointer address is followed by the size of the
	   data in use in that stack, in bytes.  This used to be a
	   conditional around just the two extra args, but that might
	   be undefined if yyoverflow is a macro.  */
	yyoverflow (YY_("memory exhausted"),
		    &yyss1, yysize * sizeof (*yyssp),
		    &yyvs1, yysize * sizeof (*yyvsp),
		    &yystacksize);

	yyss = yyss1;
	yyvs = yyvs1;
      }
#else /* no yyoverflow */
# ifndef YYSTACK_RELOCATE
      goto yyexhaustedlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
	goto yyexhaustedlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
	yystacksize = YYMAXDEPTH;

      {
	yytype_int16 *yyss1 = yyss;
	union yyalloc *yyptr =
	  (union yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (yystacksize));
	if (! yyptr)
	  goto yyexhaustedlab;
	YYSTACK_RELOCATE (yyss_alloc, yyss);
	YYSTACK_RELOCATE (yyvs_alloc, yyvs);
#  undef YYSTACK_RELOCATE
	if (yyss1 != yyssa)
	  YYSTACK_FREE (yyss1);
      }
# endif
#endif /* no yyoverflow */

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;

      YYDPRINTF ((stderr, "Stack size increased to %lu\n",
		  (unsigned long int) yystacksize));

      if (yyss + yystacksize - 1 <= yyssp)
	YYABORT;
    }

  YYDPRINTF ((stderr, "Entering state %d\n", yystate));

  if (yystate == YYFINAL)
    YYACCEPT;

  goto yybackup;

/*-----------.
| yybackup.  |
`-----------*/
yybackup:

  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  yyn = yypact[yystate];
  if (yyn == YYPACT_NINF)
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid lookahead symbol.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token: "));
      yychar = YYLEX;
    }

  if (yychar <= YYEOF)
    {
      yychar = yytoken = YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yyn == 0 || yyn == YYTABLE_NINF)
	goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);

  /* Discard the shifted token.  */
  yychar = YYEMPTY;

  yystate = yyn;
  *++yyvsp = yylval;

  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- Do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     `$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
        case 6:

/* Line 1455 of yacc.c  */
#line 224 "ptx.y"
    { set_symtab((yyvsp[(1) - (1)].ptr_value)); func_header(".skip"); ;}
    break;

  case 7:

/* Line 1455 of yacc.c  */
#line 224 "ptx.y"
    { end_function(); ;}
    break;

  case 8:

/* Line 1455 of yacc.c  */
#line 225 "ptx.y"
    { set_symtab((yyvsp[(1) - (1)].ptr_value)); ;}
    break;

  case 9:

/* Line 1455 of yacc.c  */
#line 225 "ptx.y"
    { func_header(".skip"); ;}
    break;

  case 10:

/* Line 1455 of yacc.c  */
#line 225 "ptx.y"
    { end_function(); ;}
    break;

  case 11:

/* Line 1455 of yacc.c  */
#line 228 "ptx.y"
    {func_header_info_int(".maxntid", (yyvsp[(2) - (6)].int_value));
										func_header_info_int(",", (yyvsp[(4) - (6)].int_value));
										func_header_info_int(",", (yyvsp[(6) - (6)].int_value)); ;}
    break;

  case 12:

/* Line 1455 of yacc.c  */
#line 231 "ptx.y"
    { func_header_info_int(".minnctapersm", (yyvsp[(2) - (2)].int_value)); printf("GPGPU-Sim: Warning: .minnctapersm ignored. \n"); ;}
    break;

  case 13:

/* Line 1455 of yacc.c  */
#line 232 "ptx.y"
    { func_header_info_int(".maxnctapersm", (yyvsp[(2) - (2)].int_value)); printf("GPGPU-Sim: Warning: .maxnctapersm ignored. \n"); ;}
    break;

  case 16:

/* Line 1455 of yacc.c  */
#line 239 "ptx.y"
    { start_function((yyvsp[(1) - (2)].int_value)); func_header_info("(");;}
    break;

  case 17:

/* Line 1455 of yacc.c  */
#line 239 "ptx.y"
    {func_header_info(")");;}
    break;

  case 18:

/* Line 1455 of yacc.c  */
#line 239 "ptx.y"
    { (yyval.ptr_value) = reset_symtab(); ;}
    break;

  case 19:

/* Line 1455 of yacc.c  */
#line 240 "ptx.y"
    { start_function((yyvsp[(1) - (1)].int_value)); ;}
    break;

  case 20:

/* Line 1455 of yacc.c  */
#line 240 "ptx.y"
    { (yyval.ptr_value) = reset_symtab(); ;}
    break;

  case 21:

/* Line 1455 of yacc.c  */
#line 241 "ptx.y"
    { start_function((yyvsp[(1) - (1)].int_value)); add_function_name(""); g_func_decl=0; (yyval.ptr_value) = reset_symtab(); ;}
    break;

  case 22:

/* Line 1455 of yacc.c  */
#line 244 "ptx.y"
    { add_function_name((yyvsp[(1) - (1)].string_value)); ;}
    break;

  case 23:

/* Line 1455 of yacc.c  */
#line 244 "ptx.y"
    {func_header_info("(");;}
    break;

  case 24:

/* Line 1455 of yacc.c  */
#line 244 "ptx.y"
    { g_func_decl=0; func_header_info(")"); ;}
    break;

  case 25:

/* Line 1455 of yacc.c  */
#line 245 "ptx.y"
    { add_function_name((yyvsp[(1) - (1)].string_value)); g_func_decl=0; ;}
    break;

  case 26:

/* Line 1455 of yacc.c  */
#line 248 "ptx.y"
    { (yyval.int_value) = 1; g_func_decl=1; func_header(".entry"); ;}
    break;

  case 27:

/* Line 1455 of yacc.c  */
#line 249 "ptx.y"
    { (yyval.int_value) = 1; g_func_decl=1; func_header(".entry"); ;}
    break;

  case 28:

/* Line 1455 of yacc.c  */
#line 250 "ptx.y"
    { (yyval.int_value) = 1; g_func_decl=1; func_header(".entry"); ;}
    break;

  case 29:

/* Line 1455 of yacc.c  */
#line 251 "ptx.y"
    { (yyval.int_value) = 0; g_func_decl=1; func_header(".func"); ;}
    break;

  case 30:

/* Line 1455 of yacc.c  */
#line 252 "ptx.y"
    { (yyval.int_value) = 0; g_func_decl=1; func_header(".func"); ;}
    break;

  case 31:

/* Line 1455 of yacc.c  */
#line 253 "ptx.y"
    { (yyval.int_value) = 0; g_func_decl=1; func_header(".func"); ;}
    break;

  case 32:

/* Line 1455 of yacc.c  */
#line 254 "ptx.y"
    { (yyval.int_value) = 2; g_func_decl=1; func_header(".func"); ;}
    break;

  case 33:

/* Line 1455 of yacc.c  */
#line 255 "ptx.y"
    { (yyval.int_value) = 0; g_func_decl=1; func_header(".func"); ;}
    break;

  case 35:

/* Line 1455 of yacc.c  */
#line 259 "ptx.y"
    { add_directive(); ;}
    break;

  case 36:

/* Line 1455 of yacc.c  */
#line 260 "ptx.y"
    {func_header_info(",");;}
    break;

  case 37:

/* Line 1455 of yacc.c  */
#line 260 "ptx.y"
    { add_directive(); ;}
    break;

  case 38:

/* Line 1455 of yacc.c  */
#line 262 "ptx.y"
    { add_space_spec(param_space_unclassified,0); ;}
    break;

  case 39:

/* Line 1455 of yacc.c  */
#line 262 "ptx.y"
    { add_function_arg(); ;}
    break;

  case 40:

/* Line 1455 of yacc.c  */
#line 263 "ptx.y"
    { add_space_spec(reg_space,0); ;}
    break;

  case 41:

/* Line 1455 of yacc.c  */
#line 263 "ptx.y"
    { add_function_arg(); ;}
    break;

  case 45:

/* Line 1455 of yacc.c  */
#line 269 "ptx.y"
    { add_ptr_spec(global_space); ;}
    break;

  case 46:

/* Line 1455 of yacc.c  */
#line 270 "ptx.y"
    { add_ptr_spec(local_space); ;}
    break;

  case 47:

/* Line 1455 of yacc.c  */
#line 271 "ptx.y"
    { add_ptr_spec(shared_space); ;}
    break;

  case 50:

/* Line 1455 of yacc.c  */
#line 277 "ptx.y"
    { add_directive(); ;}
    break;

  case 51:

/* Line 1455 of yacc.c  */
#line 278 "ptx.y"
    { add_instruction(); ;}
    break;

  case 52:

/* Line 1455 of yacc.c  */
#line 279 "ptx.y"
    { add_directive(); ;}
    break;

  case 53:

/* Line 1455 of yacc.c  */
#line 280 "ptx.y"
    { add_instruction(); ;}
    break;

  case 54:

/* Line 1455 of yacc.c  */
#line 281 "ptx.y"
    {start_inst_group();;}
    break;

  case 55:

/* Line 1455 of yacc.c  */
#line 281 "ptx.y"
    {end_inst_group();;}
    break;

  case 56:

/* Line 1455 of yacc.c  */
#line 282 "ptx.y"
    {start_inst_group();;}
    break;

  case 57:

/* Line 1455 of yacc.c  */
#line 282 "ptx.y"
    {end_inst_group();;}
    break;

  case 59:

/* Line 1455 of yacc.c  */
#line 286 "ptx.y"
    { add_version_info((yyvsp[(2) - (2)].double_value), 0); ;}
    break;

  case 60:

/* Line 1455 of yacc.c  */
#line 287 "ptx.y"
    { add_version_info((yyvsp[(2) - (3)].double_value),1); ;}
    break;

  case 61:

/* Line 1455 of yacc.c  */
#line 288 "ptx.y"
    {/*Do nothing*/;}
    break;

  case 62:

/* Line 1455 of yacc.c  */
#line 289 "ptx.y"
    { target_header2((yyvsp[(2) - (4)].string_value),(yyvsp[(4) - (4)].string_value)); ;}
    break;

  case 63:

/* Line 1455 of yacc.c  */
#line 290 "ptx.y"
    { target_header3((yyvsp[(2) - (6)].string_value),(yyvsp[(4) - (6)].string_value),(yyvsp[(6) - (6)].string_value)); ;}
    break;

  case 64:

/* Line 1455 of yacc.c  */
#line 291 "ptx.y"
    { target_header((yyvsp[(2) - (2)].string_value)); ;}
    break;

  case 65:

/* Line 1455 of yacc.c  */
#line 292 "ptx.y"
    { add_file((yyvsp[(2) - (3)].int_value),(yyvsp[(3) - (3)].string_value)); ;}
    break;

  case 66:

/* Line 1455 of yacc.c  */
#line 293 "ptx.y"
    { add_file((yyvsp[(2) - (7)].int_value),(yyvsp[(3) - (7)].string_value)); ;}
    break;

  case 68:

/* Line 1455 of yacc.c  */
#line 295 "ptx.y"
    { add_pragma((yyvsp[(2) - (3)].string_value)); ;}
    break;

  case 69:

/* Line 1455 of yacc.c  */
#line 296 "ptx.y"
    {/*Do nothing*/;}
    break;

  case 70:

/* Line 1455 of yacc.c  */
#line 299 "ptx.y"
    { add_variables(); ;}
    break;

  case 71:

/* Line 1455 of yacc.c  */
#line 300 "ptx.y"
    { add_variables(); ;}
    break;

  case 72:

/* Line 1455 of yacc.c  */
#line 301 "ptx.y"
    { add_variables(); ;}
    break;

  case 73:

/* Line 1455 of yacc.c  */
#line 302 "ptx.y"
    { add_constptr((yyvsp[(2) - (6)].string_value), (yyvsp[(4) - (6)].string_value), (yyvsp[(6) - (6)].int_value)); ;}
    break;

  case 74:

/* Line 1455 of yacc.c  */
#line 305 "ptx.y"
    { set_variable_type(); ;}
    break;

  case 77:

/* Line 1455 of yacc.c  */
#line 310 "ptx.y"
    { add_identifier((yyvsp[(1) - (1)].string_value),0,NON_ARRAY_IDENTIFIER); func_header_info((yyvsp[(1) - (1)].string_value));;}
    break;

  case 78:

/* Line 1455 of yacc.c  */
#line 311 "ptx.y"
    { func_header_info((yyvsp[(1) - (4)].string_value)); func_header_info_int("<", (yyvsp[(3) - (4)].int_value)); func_header_info(">");
		int i,lbase,l;
		char *id = NULL;
		lbase = strlen((yyvsp[(1) - (4)].string_value));
		for( i=0; i < (yyvsp[(3) - (4)].int_value); i++ ) { 
			l = lbase + (int)log10(i+1)+10;
			id = (char*) malloc(l);
			snprintf(id,l,"%s%u",(yyvsp[(1) - (4)].string_value),i);
			add_identifier(id,0,NON_ARRAY_IDENTIFIER); 
		}
		free((yyvsp[(1) - (4)].string_value));
	;}
    break;

  case 79:

/* Line 1455 of yacc.c  */
#line 323 "ptx.y"
    { add_identifier((yyvsp[(1) - (3)].string_value),0,ARRAY_IDENTIFIER_NO_DIM); func_header_info((yyvsp[(1) - (3)].string_value)); func_header_info("["); func_header_info("]");;}
    break;

  case 80:

/* Line 1455 of yacc.c  */
#line 324 "ptx.y"
    { add_identifier((yyvsp[(1) - (4)].string_value),(yyvsp[(3) - (4)].int_value),ARRAY_IDENTIFIER); func_header_info((yyvsp[(1) - (4)].string_value)); func_header_info_int("[",(yyvsp[(3) - (4)].int_value)); func_header_info("]");;}
    break;

  case 86:

/* Line 1455 of yacc.c  */
#line 333 "ptx.y"
    { add_extern_spec(); ;}
    break;

  case 88:

/* Line 1455 of yacc.c  */
#line 337 "ptx.y"
    { add_alignment_spec((yyvsp[(2) - (2)].int_value)); ;}
    break;

  case 89:

/* Line 1455 of yacc.c  */
#line 339 "ptx.y"
    {  add_space_spec(reg_space,0); ;}
    break;

  case 90:

/* Line 1455 of yacc.c  */
#line 340 "ptx.y"
    {  add_space_spec(reg_space,0); ;}
    break;

  case 92:

/* Line 1455 of yacc.c  */
#line 344 "ptx.y"
    {  add_space_spec(const_space,(yyvsp[(1) - (1)].int_value)); ;}
    break;

  case 93:

/* Line 1455 of yacc.c  */
#line 345 "ptx.y"
    {  add_space_spec(global_space,0); ;}
    break;

  case 94:

/* Line 1455 of yacc.c  */
#line 346 "ptx.y"
    {  add_space_spec(local_space,0); ;}
    break;

  case 95:

/* Line 1455 of yacc.c  */
#line 347 "ptx.y"
    {  add_space_spec(param_space_unclassified,0); ;}
    break;

  case 96:

/* Line 1455 of yacc.c  */
#line 348 "ptx.y"
    {  add_space_spec(shared_space,0); ;}
    break;

  case 97:

/* Line 1455 of yacc.c  */
#line 349 "ptx.y"
    {  add_space_spec(surf_space,0); ;}
    break;

  case 98:

/* Line 1455 of yacc.c  */
#line 350 "ptx.y"
    {  add_space_spec(tex_space,0); ;}
    break;

  case 101:

/* Line 1455 of yacc.c  */
#line 357 "ptx.y"
    {  add_option(V2_TYPE); func_header_info(".v2");;}
    break;

  case 102:

/* Line 1455 of yacc.c  */
#line 358 "ptx.y"
    {  add_option(V3_TYPE); func_header_info(".v3");;}
    break;

  case 103:

/* Line 1455 of yacc.c  */
#line 359 "ptx.y"
    {  add_option(V4_TYPE); func_header_info(".v4");;}
    break;

  case 104:

/* Line 1455 of yacc.c  */
#line 362 "ptx.y"
    { add_scalar_type_spec( S8_TYPE ); ;}
    break;

  case 105:

/* Line 1455 of yacc.c  */
#line 363 "ptx.y"
    { add_scalar_type_spec( S16_TYPE ); ;}
    break;

  case 106:

/* Line 1455 of yacc.c  */
#line 364 "ptx.y"
    { add_scalar_type_spec( S32_TYPE ); ;}
    break;

  case 107:

/* Line 1455 of yacc.c  */
#line 365 "ptx.y"
    { add_scalar_type_spec( S64_TYPE ); ;}
    break;

  case 108:

/* Line 1455 of yacc.c  */
#line 366 "ptx.y"
    { add_scalar_type_spec( U8_TYPE ); ;}
    break;

  case 109:

/* Line 1455 of yacc.c  */
#line 367 "ptx.y"
    { add_scalar_type_spec( U16_TYPE ); ;}
    break;

  case 110:

/* Line 1455 of yacc.c  */
#line 368 "ptx.y"
    { add_scalar_type_spec( U32_TYPE ); ;}
    break;

  case 111:

/* Line 1455 of yacc.c  */
#line 369 "ptx.y"
    { add_scalar_type_spec( U64_TYPE ); ;}
    break;

  case 112:

/* Line 1455 of yacc.c  */
#line 370 "ptx.y"
    { add_scalar_type_spec( F16_TYPE ); ;}
    break;

  case 113:

/* Line 1455 of yacc.c  */
#line 371 "ptx.y"
    { add_scalar_type_spec( F32_TYPE ); ;}
    break;

  case 114:

/* Line 1455 of yacc.c  */
#line 372 "ptx.y"
    { add_scalar_type_spec( F64_TYPE ); ;}
    break;

  case 115:

/* Line 1455 of yacc.c  */
#line 373 "ptx.y"
    { add_scalar_type_spec( FF64_TYPE ); ;}
    break;

  case 116:

/* Line 1455 of yacc.c  */
#line 374 "ptx.y"
    { add_scalar_type_spec( B8_TYPE );  ;}
    break;

  case 117:

/* Line 1455 of yacc.c  */
#line 375 "ptx.y"
    { add_scalar_type_spec( B16_TYPE ); ;}
    break;

  case 118:

/* Line 1455 of yacc.c  */
#line 376 "ptx.y"
    { add_scalar_type_spec( B32_TYPE ); ;}
    break;

  case 119:

/* Line 1455 of yacc.c  */
#line 377 "ptx.y"
    { add_scalar_type_spec( B64_TYPE ); ;}
    break;

  case 120:

/* Line 1455 of yacc.c  */
#line 378 "ptx.y"
    { add_scalar_type_spec( BB64_TYPE ); ;}
    break;

  case 121:

/* Line 1455 of yacc.c  */
#line 379 "ptx.y"
    { add_scalar_type_spec( BB128_TYPE ); ;}
    break;

  case 122:

/* Line 1455 of yacc.c  */
#line 380 "ptx.y"
    { add_scalar_type_spec( PRED_TYPE ); ;}
    break;

  case 123:

/* Line 1455 of yacc.c  */
#line 381 "ptx.y"
    { add_scalar_type_spec( TEXREF_TYPE ); ;}
    break;

  case 124:

/* Line 1455 of yacc.c  */
#line 382 "ptx.y"
    { add_scalar_type_spec( SAMPLERREF_TYPE ); ;}
    break;

  case 125:

/* Line 1455 of yacc.c  */
#line 383 "ptx.y"
    { add_scalar_type_spec( SURFREF_TYPE ); ;}
    break;

  case 126:

/* Line 1455 of yacc.c  */
#line 386 "ptx.y"
    { add_array_initializer(); ;}
    break;

  case 127:

/* Line 1455 of yacc.c  */
#line 387 "ptx.y"
    { syntax_not_implemented(); ;}
    break;

  case 131:

/* Line 1455 of yacc.c  */
#line 393 "ptx.y"
    { add_label((yyvsp[(1) - (2)].string_value)); ;}
    break;

  case 133:

/* Line 1455 of yacc.c  */
#line 396 "ptx.y"
    { set_return(); ;}
    break;

  case 139:

/* Line 1455 of yacc.c  */
#line 403 "ptx.y"
    { add_opcode((yyvsp[(1) - (1)].int_value)); ;}
    break;

  case 141:

/* Line 1455 of yacc.c  */
#line 404 "ptx.y"
    { add_opcode((yyvsp[(1) - (1)].int_value)); ;}
    break;

  case 142:

/* Line 1455 of yacc.c  */
#line 406 "ptx.y"
    { add_pred((yyvsp[(2) - (2)].string_value),0, -1); ;}
    break;

  case 143:

/* Line 1455 of yacc.c  */
#line 407 "ptx.y"
    { add_pred((yyvsp[(3) - (3)].string_value),1, -1); ;}
    break;

  case 144:

/* Line 1455 of yacc.c  */
#line 408 "ptx.y"
    { add_pred((yyvsp[(2) - (3)].string_value),0,1); ;}
    break;

  case 145:

/* Line 1455 of yacc.c  */
#line 409 "ptx.y"
    { add_pred((yyvsp[(2) - (3)].string_value),0,2); ;}
    break;

  case 146:

/* Line 1455 of yacc.c  */
#line 410 "ptx.y"
    { add_pred((yyvsp[(2) - (3)].string_value),0,3); ;}
    break;

  case 147:

/* Line 1455 of yacc.c  */
#line 411 "ptx.y"
    { add_pred((yyvsp[(2) - (3)].string_value),0,5); ;}
    break;

  case 148:

/* Line 1455 of yacc.c  */
#line 412 "ptx.y"
    { add_pred((yyvsp[(2) - (3)].string_value),0,6); ;}
    break;

  case 149:

/* Line 1455 of yacc.c  */
#line 413 "ptx.y"
    { add_pred((yyvsp[(2) - (3)].string_value),0,10); ;}
    break;

  case 150:

/* Line 1455 of yacc.c  */
#line 414 "ptx.y"
    { add_pred((yyvsp[(2) - (3)].string_value),0,12); ;}
    break;

  case 151:

/* Line 1455 of yacc.c  */
#line 415 "ptx.y"
    { add_pred((yyvsp[(2) - (3)].string_value),0,13); ;}
    break;

  case 152:

/* Line 1455 of yacc.c  */
#line 416 "ptx.y"
    { add_pred((yyvsp[(2) - (3)].string_value),0,17); ;}
    break;

  case 153:

/* Line 1455 of yacc.c  */
#line 417 "ptx.y"
    { add_pred((yyvsp[(2) - (3)].string_value),0,19); ;}
    break;

  case 154:

/* Line 1455 of yacc.c  */
#line 418 "ptx.y"
    { add_pred((yyvsp[(2) - (3)].string_value),0,28); ;}
    break;

  case 161:

/* Line 1455 of yacc.c  */
#line 428 "ptx.y"
    { add_option(SYNC_OPTION); ;}
    break;

  case 162:

/* Line 1455 of yacc.c  */
#line 429 "ptx.y"
    { add_option(ARRIVE_OPTION); ;}
    break;

  case 163:

/* Line 1455 of yacc.c  */
#line 430 "ptx.y"
    { add_option(RED_OPTION); ;}
    break;

  case 164:

/* Line 1455 of yacc.c  */
#line 431 "ptx.y"
    { add_option(UNI_OPTION); ;}
    break;

  case 165:

/* Line 1455 of yacc.c  */
#line 432 "ptx.y"
    { add_option(WIDE_OPTION); ;}
    break;

  case 166:

/* Line 1455 of yacc.c  */
#line 433 "ptx.y"
    { add_option(ANY_OPTION); ;}
    break;

  case 167:

/* Line 1455 of yacc.c  */
#line 434 "ptx.y"
    { add_option(ALL_OPTION); ;}
    break;

  case 168:

/* Line 1455 of yacc.c  */
#line 435 "ptx.y"
    { add_option(BALLOT_OPTION); ;}
    break;

  case 169:

/* Line 1455 of yacc.c  */
#line 436 "ptx.y"
    { add_option(GLOBAL_OPTION); ;}
    break;

  case 170:

/* Line 1455 of yacc.c  */
#line 437 "ptx.y"
    { add_option(CTA_OPTION); ;}
    break;

  case 171:

/* Line 1455 of yacc.c  */
#line 438 "ptx.y"
    { add_option(SYS_OPTION); ;}
    break;

  case 172:

/* Line 1455 of yacc.c  */
#line 439 "ptx.y"
    { add_option(GEOM_MODIFIER_1D); ;}
    break;

  case 173:

/* Line 1455 of yacc.c  */
#line 440 "ptx.y"
    { add_option(GEOM_MODIFIER_2D); ;}
    break;

  case 174:

/* Line 1455 of yacc.c  */
#line 441 "ptx.y"
    { add_option(GEOM_MODIFIER_3D); ;}
    break;

  case 175:

/* Line 1455 of yacc.c  */
#line 442 "ptx.y"
    { add_option(SAT_OPTION); ;}
    break;

  case 176:

/* Line 1455 of yacc.c  */
#line 443 "ptx.y"
    { add_option(FTZ_OPTION); ;}
    break;

  case 177:

/* Line 1455 of yacc.c  */
#line 444 "ptx.y"
    { add_option(NEG_OPTION); ;}
    break;

  case 178:

/* Line 1455 of yacc.c  */
#line 445 "ptx.y"
    { add_option(APPROX_OPTION); ;}
    break;

  case 179:

/* Line 1455 of yacc.c  */
#line 446 "ptx.y"
    { add_option(FULL_OPTION); ;}
    break;

  case 180:

/* Line 1455 of yacc.c  */
#line 447 "ptx.y"
    { add_option(EXIT_OPTION); ;}
    break;

  case 181:

/* Line 1455 of yacc.c  */
#line 448 "ptx.y"
    { add_option(ABS_OPTION); ;}
    break;

  case 183:

/* Line 1455 of yacc.c  */
#line 450 "ptx.y"
    { add_option(TO_OPTION); ;}
    break;

  case 184:

/* Line 1455 of yacc.c  */
#line 451 "ptx.y"
    { add_option(HALF_OPTION); ;}
    break;

  case 185:

/* Line 1455 of yacc.c  */
#line 452 "ptx.y"
    { add_option(EXTP_OPTION); ;}
    break;

  case 186:

/* Line 1455 of yacc.c  */
#line 453 "ptx.y"
    { add_option(CA_OPTION); ;}
    break;

  case 187:

/* Line 1455 of yacc.c  */
#line 454 "ptx.y"
    { add_option(CG_OPTION); ;}
    break;

  case 188:

/* Line 1455 of yacc.c  */
#line 455 "ptx.y"
    { add_option(CS_OPTION); ;}
    break;

  case 189:

/* Line 1455 of yacc.c  */
#line 456 "ptx.y"
    { add_option(LU_OPTION); ;}
    break;

  case 190:

/* Line 1455 of yacc.c  */
#line 457 "ptx.y"
    { add_option(CV_OPTION); ;}
    break;

  case 191:

/* Line 1455 of yacc.c  */
#line 458 "ptx.y"
    { add_option(WB_OPTION); ;}
    break;

  case 192:

/* Line 1455 of yacc.c  */
#line 459 "ptx.y"
    { add_option(WT_OPTION); ;}
    break;

  case 193:

/* Line 1455 of yacc.c  */
#line 460 "ptx.y"
    { add_option(NC_OPTION); ;}
    break;

  case 194:

/* Line 1455 of yacc.c  */
#line 461 "ptx.y"
    { add_option(UP_OPTION); ;}
    break;

  case 195:

/* Line 1455 of yacc.c  */
#line 462 "ptx.y"
    { add_option(DOWN_OPTION); ;}
    break;

  case 196:

/* Line 1455 of yacc.c  */
#line 463 "ptx.y"
    { add_option(BFLY_OPTION); ;}
    break;

  case 197:

/* Line 1455 of yacc.c  */
#line 464 "ptx.y"
    { add_option(IDX_OPTION); ;}
    break;

  case 198:

/* Line 1455 of yacc.c  */
#line 467 "ptx.y"
    { add_option(ATOMIC_AND); ;}
    break;

  case 199:

/* Line 1455 of yacc.c  */
#line 468 "ptx.y"
    { add_option(ATOMIC_POPC); ;}
    break;

  case 200:

/* Line 1455 of yacc.c  */
#line 469 "ptx.y"
    { add_option(ATOMIC_OR); ;}
    break;

  case 201:

/* Line 1455 of yacc.c  */
#line 470 "ptx.y"
    { add_option(ATOMIC_XOR); ;}
    break;

  case 202:

/* Line 1455 of yacc.c  */
#line 471 "ptx.y"
    { add_option(ATOMIC_CAS); ;}
    break;

  case 203:

/* Line 1455 of yacc.c  */
#line 472 "ptx.y"
    { add_option(ATOMIC_EXCH); ;}
    break;

  case 204:

/* Line 1455 of yacc.c  */
#line 473 "ptx.y"
    { add_option(ATOMIC_ADD); ;}
    break;

  case 205:

/* Line 1455 of yacc.c  */
#line 474 "ptx.y"
    { add_option(ATOMIC_INC); ;}
    break;

  case 206:

/* Line 1455 of yacc.c  */
#line 475 "ptx.y"
    { add_option(ATOMIC_DEC); ;}
    break;

  case 207:

/* Line 1455 of yacc.c  */
#line 476 "ptx.y"
    { add_option(ATOMIC_MIN); ;}
    break;

  case 208:

/* Line 1455 of yacc.c  */
#line 477 "ptx.y"
    { add_option(ATOMIC_MAX); ;}
    break;

  case 211:

/* Line 1455 of yacc.c  */
#line 483 "ptx.y"
    { add_option(RN_OPTION); ;}
    break;

  case 212:

/* Line 1455 of yacc.c  */
#line 484 "ptx.y"
    { add_option(RZ_OPTION); ;}
    break;

  case 213:

/* Line 1455 of yacc.c  */
#line 485 "ptx.y"
    { add_option(RM_OPTION); ;}
    break;

  case 214:

/* Line 1455 of yacc.c  */
#line 486 "ptx.y"
    { add_option(RP_OPTION); ;}
    break;

  case 215:

/* Line 1455 of yacc.c  */
#line 489 "ptx.y"
    { add_option(RNI_OPTION); ;}
    break;

  case 216:

/* Line 1455 of yacc.c  */
#line 490 "ptx.y"
    { add_option(RZI_OPTION); ;}
    break;

  case 217:

/* Line 1455 of yacc.c  */
#line 491 "ptx.y"
    { add_option(RMI_OPTION); ;}
    break;

  case 218:

/* Line 1455 of yacc.c  */
#line 492 "ptx.y"
    { add_option(RPI_OPTION); ;}
    break;

  case 219:

/* Line 1455 of yacc.c  */
#line 495 "ptx.y"
    { add_option(EQ_OPTION); ;}
    break;

  case 220:

/* Line 1455 of yacc.c  */
#line 496 "ptx.y"
    { add_option(NE_OPTION); ;}
    break;

  case 221:

/* Line 1455 of yacc.c  */
#line 497 "ptx.y"
    { add_option(LT_OPTION); ;}
    break;

  case 222:

/* Line 1455 of yacc.c  */
#line 498 "ptx.y"
    { add_option(LE_OPTION); ;}
    break;

  case 223:

/* Line 1455 of yacc.c  */
#line 499 "ptx.y"
    { add_option(GT_OPTION); ;}
    break;

  case 224:

/* Line 1455 of yacc.c  */
#line 500 "ptx.y"
    { add_option(GE_OPTION); ;}
    break;

  case 225:

/* Line 1455 of yacc.c  */
#line 501 "ptx.y"
    { add_option(LO_OPTION); ;}
    break;

  case 226:

/* Line 1455 of yacc.c  */
#line 502 "ptx.y"
    { add_option(LS_OPTION); ;}
    break;

  case 227:

/* Line 1455 of yacc.c  */
#line 503 "ptx.y"
    { add_option(HI_OPTION); ;}
    break;

  case 228:

/* Line 1455 of yacc.c  */
#line 504 "ptx.y"
    { add_option(HS_OPTION); ;}
    break;

  case 229:

/* Line 1455 of yacc.c  */
#line 505 "ptx.y"
    { add_option(EQU_OPTION); ;}
    break;

  case 230:

/* Line 1455 of yacc.c  */
#line 506 "ptx.y"
    { add_option(NEU_OPTION); ;}
    break;

  case 231:

/* Line 1455 of yacc.c  */
#line 507 "ptx.y"
    { add_option(LTU_OPTION); ;}
    break;

  case 232:

/* Line 1455 of yacc.c  */
#line 508 "ptx.y"
    { add_option(LEU_OPTION); ;}
    break;

  case 233:

/* Line 1455 of yacc.c  */
#line 509 "ptx.y"
    { add_option(GTU_OPTION); ;}
    break;

  case 234:

/* Line 1455 of yacc.c  */
#line 510 "ptx.y"
    { add_option(GEU_OPTION); ;}
    break;

  case 235:

/* Line 1455 of yacc.c  */
#line 511 "ptx.y"
    { add_option(NUM_OPTION); ;}
    break;

  case 236:

/* Line 1455 of yacc.c  */
#line 512 "ptx.y"
    { add_option(NAN_OPTION); ;}
    break;

  case 240:

/* Line 1455 of yacc.c  */
#line 519 "ptx.y"
    { add_scalar_operand( (yyvsp[(1) - (1)].string_value) ); ;}
    break;

  case 241:

/* Line 1455 of yacc.c  */
#line 520 "ptx.y"
    { add_neg_pred_operand( (yyvsp[(2) - (2)].string_value) ); ;}
    break;

  case 242:

/* Line 1455 of yacc.c  */
#line 521 "ptx.y"
    { add_scalar_operand( (yyvsp[(2) - (2)].string_value) ); change_operand_neg(); ;}
    break;

  case 247:

/* Line 1455 of yacc.c  */
#line 526 "ptx.y"
    { change_operand_neg(); ;}
    break;

  case 249:

/* Line 1455 of yacc.c  */
#line 528 "ptx.y"
    { add_address_operand((yyvsp[(1) - (3)].string_value),(yyvsp[(3) - (3)].int_value)); ;}
    break;

  case 250:

/* Line 1455 of yacc.c  */
#line 529 "ptx.y"
    { add_scalar_operand( (yyvsp[(1) - (2)].string_value) ); change_operand_lohi(1);;}
    break;

  case 251:

/* Line 1455 of yacc.c  */
#line 530 "ptx.y"
    { add_scalar_operand( (yyvsp[(2) - (3)].string_value) ); change_operand_lohi(1); change_operand_neg();;}
    break;

  case 252:

/* Line 1455 of yacc.c  */
#line 531 "ptx.y"
    { add_scalar_operand( (yyvsp[(1) - (2)].string_value) ); change_operand_lohi(2);;}
    break;

  case 253:

/* Line 1455 of yacc.c  */
#line 532 "ptx.y"
    { add_scalar_operand( (yyvsp[(2) - (3)].string_value) ); change_operand_lohi(2); change_operand_neg();;}
    break;

  case 254:

/* Line 1455 of yacc.c  */
#line 533 "ptx.y"
    { add_2vector_operand((yyvsp[(1) - (3)].string_value),(yyvsp[(3) - (3)].string_value)); change_double_operand_type(-1);;}
    break;

  case 255:

/* Line 1455 of yacc.c  */
#line 534 "ptx.y"
    { add_2vector_operand((yyvsp[(1) - (4)].string_value),(yyvsp[(3) - (4)].string_value)); change_double_operand_type(-1); change_operand_lohi(1);;}
    break;

  case 256:

/* Line 1455 of yacc.c  */
#line 535 "ptx.y"
    { add_2vector_operand((yyvsp[(1) - (4)].string_value),(yyvsp[(3) - (4)].string_value)); change_double_operand_type(-1); change_operand_lohi(2);;}
    break;

  case 257:

/* Line 1455 of yacc.c  */
#line 536 "ptx.y"
    { add_2vector_operand((yyvsp[(1) - (3)].string_value),(yyvsp[(3) - (3)].string_value)); change_double_operand_type(-3);;}
    break;

  case 258:

/* Line 1455 of yacc.c  */
#line 537 "ptx.y"
    { add_2vector_operand((yyvsp[(1) - (4)].string_value),(yyvsp[(3) - (4)].string_value)); change_double_operand_type(-3); change_operand_lohi(1);;}
    break;

  case 259:

/* Line 1455 of yacc.c  */
#line 538 "ptx.y"
    { add_2vector_operand((yyvsp[(1) - (4)].string_value),(yyvsp[(3) - (4)].string_value)); change_double_operand_type(-3); change_operand_lohi(2);;}
    break;

  case 260:

/* Line 1455 of yacc.c  */
#line 541 "ptx.y"
    { add_2vector_operand((yyvsp[(2) - (5)].string_value),(yyvsp[(4) - (5)].string_value)); ;}
    break;

  case 261:

/* Line 1455 of yacc.c  */
#line 542 "ptx.y"
    { add_3vector_operand((yyvsp[(2) - (7)].string_value),(yyvsp[(4) - (7)].string_value),(yyvsp[(6) - (7)].string_value)); ;}
    break;

  case 262:

/* Line 1455 of yacc.c  */
#line 543 "ptx.y"
    { add_4vector_operand((yyvsp[(2) - (9)].string_value),(yyvsp[(4) - (9)].string_value),(yyvsp[(6) - (9)].string_value),(yyvsp[(8) - (9)].string_value)); ;}
    break;

  case 263:

/* Line 1455 of yacc.c  */
#line 544 "ptx.y"
    { add_1vector_operand((yyvsp[(2) - (3)].string_value)); ;}
    break;

  case 264:

/* Line 1455 of yacc.c  */
#line 547 "ptx.y"
    { add_scalar_operand((yyvsp[(2) - (3)].string_value)); ;}
    break;

  case 266:

/* Line 1455 of yacc.c  */
#line 552 "ptx.y"
    { add_builtin_operand((yyvsp[(1) - (2)].int_value),(yyvsp[(2) - (2)].int_value)); ;}
    break;

  case 267:

/* Line 1455 of yacc.c  */
#line 553 "ptx.y"
    { add_builtin_operand((yyvsp[(1) - (1)].int_value),-1); ;}
    break;

  case 268:

/* Line 1455 of yacc.c  */
#line 556 "ptx.y"
    { add_memory_operand(); ;}
    break;

  case 269:

/* Line 1455 of yacc.c  */
#line 557 "ptx.y"
    { add_memory_operand(); change_memory_addr_space((yyvsp[(1) - (4)].string_value)); ;}
    break;

  case 270:

/* Line 1455 of yacc.c  */
#line 558 "ptx.y"
    { change_memory_addr_space((yyvsp[(1) - (4)].string_value)); ;}
    break;

  case 271:

/* Line 1455 of yacc.c  */
#line 559 "ptx.y"
    { change_memory_addr_space((yyvsp[(1) - (4)].string_value)); add_memory_operand();;}
    break;

  case 272:

/* Line 1455 of yacc.c  */
#line 560 "ptx.y"
    { change_operand_neg(); ;}
    break;

  case 273:

/* Line 1455 of yacc.c  */
#line 563 "ptx.y"
    { add_double_operand((yyvsp[(1) - (3)].string_value),(yyvsp[(3) - (3)].string_value)); change_double_operand_type(1); ;}
    break;

  case 274:

/* Line 1455 of yacc.c  */
#line 564 "ptx.y"
    { add_double_operand((yyvsp[(1) - (4)].string_value),(yyvsp[(3) - (4)].string_value)); change_double_operand_type(1); change_operand_lohi(1); ;}
    break;

  case 275:

/* Line 1455 of yacc.c  */
#line 565 "ptx.y"
    { add_double_operand((yyvsp[(1) - (4)].string_value),(yyvsp[(3) - (4)].string_value)); change_double_operand_type(1); change_operand_lohi(2); ;}
    break;

  case 276:

/* Line 1455 of yacc.c  */
#line 566 "ptx.y"
    { add_double_operand((yyvsp[(1) - (4)].string_value),(yyvsp[(4) - (4)].string_value)); change_double_operand_type(2); ;}
    break;

  case 277:

/* Line 1455 of yacc.c  */
#line 567 "ptx.y"
    { add_double_operand((yyvsp[(1) - (5)].string_value),(yyvsp[(4) - (5)].string_value)); change_double_operand_type(2); change_operand_lohi(1); ;}
    break;

  case 278:

/* Line 1455 of yacc.c  */
#line 568 "ptx.y"
    { add_double_operand((yyvsp[(1) - (5)].string_value),(yyvsp[(4) - (5)].string_value)); change_double_operand_type(2); change_operand_lohi(2); ;}
    break;

  case 279:

/* Line 1455 of yacc.c  */
#line 569 "ptx.y"
    { add_address_operand((yyvsp[(1) - (4)].string_value),(yyvsp[(4) - (4)].int_value)); change_double_operand_type(3); ;}
    break;

  case 280:

/* Line 1455 of yacc.c  */
#line 572 "ptx.y"
    { add_literal_int((yyvsp[(1) - (1)].int_value)); ;}
    break;

  case 281:

/* Line 1455 of yacc.c  */
#line 573 "ptx.y"
    { add_literal_float((yyvsp[(1) - (1)].float_value)); ;}
    break;

  case 282:

/* Line 1455 of yacc.c  */
#line 574 "ptx.y"
    { add_literal_double((yyvsp[(1) - (1)].double_value)); ;}
    break;

  case 283:

/* Line 1455 of yacc.c  */
#line 577 "ptx.y"
    { add_address_operand((yyvsp[(1) - (1)].string_value),0); ;}
    break;

  case 284:

/* Line 1455 of yacc.c  */
#line 578 "ptx.y"
    { add_address_operand((yyvsp[(1) - (2)].string_value),0); change_operand_lohi(1);;}
    break;

  case 285:

/* Line 1455 of yacc.c  */
#line 579 "ptx.y"
    { add_address_operand((yyvsp[(1) - (2)].string_value),0); change_operand_lohi(2); ;}
    break;

  case 286:

/* Line 1455 of yacc.c  */
#line 580 "ptx.y"
    { add_address_operand((yyvsp[(1) - (3)].string_value),(yyvsp[(3) - (3)].int_value)); ;}
    break;

  case 287:

/* Line 1455 of yacc.c  */
#line 581 "ptx.y"
    { add_address_operand2((yyvsp[(1) - (1)].int_value)); ;}
    break;



/* Line 1455 of yacc.c  */
#line 3658 "/home/j/code/gpgpu-sim_distribution-dev/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c"
      default: break;
    }
  YY_SYMBOL_PRINT ("-> $$ =", yyr1[yyn], &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);

  *++yyvsp = yyval;

  /* Now `shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTOKENS] + *yyssp;
  if (0 <= yystate && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTOKENS];

  goto yynewstate;


/*------------------------------------.
| yyerrlab -- here on detecting error |
`------------------------------------*/
yyerrlab:
  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
#if ! YYERROR_VERBOSE
      yyerror (YY_("syntax error"));
#else
      {
	YYSIZE_T yysize = yysyntax_error (0, yystate, yychar);
	if (yymsg_alloc < yysize && yymsg_alloc < YYSTACK_ALLOC_MAXIMUM)
	  {
	    YYSIZE_T yyalloc = 2 * yysize;
	    if (! (yysize <= yyalloc && yyalloc <= YYSTACK_ALLOC_MAXIMUM))
	      yyalloc = YYSTACK_ALLOC_MAXIMUM;
	    if (yymsg != yymsgbuf)
	      YYSTACK_FREE (yymsg);
	    yymsg = (char *) YYSTACK_ALLOC (yyalloc);
	    if (yymsg)
	      yymsg_alloc = yyalloc;
	    else
	      {
		yymsg = yymsgbuf;
		yymsg_alloc = sizeof yymsgbuf;
	      }
	  }

	if (0 < yysize && yysize <= yymsg_alloc)
	  {
	    (void) yysyntax_error (yymsg, yystate, yychar);
	    yyerror (yymsg);
	  }
	else
	  {
	    yyerror (YY_("syntax error"));
	    if (yysize != 0)
	      goto yyexhaustedlab;
	  }
      }
#endif
    }



  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
	 error, discard it.  */

      if (yychar <= YYEOF)
	{
	  /* Return failure if at end of input.  */
	  if (yychar == YYEOF)
	    YYABORT;
	}
      else
	{
	  yydestruct ("Error: discarding",
		      yytoken, &yylval);
	  yychar = YYEMPTY;
	}
    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:

  /* Pacify compilers like GCC when the user code never invokes
     YYERROR and the label yyerrorlab therefore never appears in user
     code.  */
  if (/*CONSTCOND*/ 0)
     goto yyerrorlab;

  /* Do not reclaim the symbols of the rule which action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;	/* Each real token shifted decrements this.  */

  for (;;)
    {
      yyn = yypact[yystate];
      if (yyn != YYPACT_NINF)
	{
	  yyn += YYTERROR;
	  if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR)
	    {
	      yyn = yytable[yyn];
	      if (0 < yyn)
		break;
	    }
	}

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
	YYABORT;


      yydestruct ("Error: popping",
		  yystos[yystate], yyvsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  *++yyvsp = yylval;


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", yystos[yyn], yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;

/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;

#if !defined(yyoverflow) || YYERROR_VERBOSE
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
  if (yychar != YYEMPTY)
     yydestruct ("Cleanup: discarding lookahead",
		 yytoken, &yylval);
  /* Do not reclaim the symbols of the rule which action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
		  yystos[*yyssp], yyvsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
#if YYERROR_VERBOSE
  if (yymsg != yymsgbuf)
    YYSTACK_FREE (yymsg);
#endif
  /* Make sure YYID is used.  */
  return YYID (yyresult);
}



/* Line 1675 of yacc.c  */
#line 584 "ptx.y"


extern int ptx_lineno;
extern const char *g_filename;

void syntax_not_implemented()
{
	printf("Parse error (%s:%u): this syntax is not (yet) implemented:\n",g_filename,ptx_lineno);
	ptx_error(NULL);
	abort();
}

