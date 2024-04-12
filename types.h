#ifndef __TYPES_H__
#define __TYPES_H__

typedef short vim_int16;
typedef unsigned short vim_uint16;
typedef char  vim_int8;
typedef unsigned char vim_uint8;
typedef long vim_int32;
typedef long long vim_int64;
typedef unsigned long vim_uint32;
typedef unsigned long long vim_uint64;
typedef float         vim_float32;
typedef double        vim_float64;


enum DTYPE {ENUM_INT8 = 0, ENUM_UINT8, ENUM_INT16, ENUM_UINT16, ENUM_INT32, ENUM_UINT32, EMUM_REAL, ENUM_FLOAT64};

const vim_uint32 DTypeLen[] = {1, 1, 2, 2, 4, 4, 4, 8}; 

#define Real vim_float32
#define EMUM_REAL (EMUM_REAL)

typedef struct _HW{
    vim_uint32 h;
    vim_uint32 w;
}HW;

#endif