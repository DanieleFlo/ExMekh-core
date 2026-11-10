#include <math.h>
#include <stddef.h>
void position(const float *values, const float *weights, float *output, size_t count) {
 for(size_t i=0;i<count;i++)output[i] = values[i*4]*weights[i*4]+values[i*4+1]*weights[i*4+1]+values[i*4+2]*weights[i*4+2]+values[i*4+3]*weights[i*4+3];
}
void unfused_position(const float *values, const float *weights, float *output, size_t count) {
#pragma clang fp contract(off)
 for(size_t i=0;i<count;i++)output[i] = values[i*4]*weights[i*4]+values[i*4+1]*weights[i*4+1]+values[i*4+2]*weights[i*4+2]+values[i*4+3]*weights[i*4+3];
}
void frequency(float *out, float *cosines, float *sines) {
 for(unsigned coord=0;coord<16;coord++)for(unsigned d=0;d<18;d++) {
  unsigned i=coord*18+d;
  float x=coord*(1.0f/powf(10000.0f,(float)(2*d)/36));
  out[i]=x;cosines[i]=cosf(x);sines[i]=sinf(x);
 }
}
