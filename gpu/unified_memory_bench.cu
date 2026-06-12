// SMASH Phase 4 — GB10 unified-memory benchmark.
//
// The main prototype measured the GPU path with explicit cudaMemcpy, which is
// the *discrete-GPU* model and overstates the cost on a coherent unified-memory
// part. This benchmark re-measures the same propagation+finding work end-to-end
// (CPU produces input -> GPU computes -> CPU consumes output) under the memory
// models actually available on GB10 (Grace-Blackwell, ATS coherent):
//
//   [CPU]        OpenMP baseline.
//   [explicit]   cudaMalloc + cudaMemcpy H2D/D2H (discrete-GPU model).
//   [managed]    cudaMallocManaged, on-demand page migration.
//   [managed+pf] cudaMallocManaged + cudaMemPrefetchAsync.
//   [ATS/malloc] plain malloc passed straight to the kernel (GB10 coherence).
//   [resident]   data already on the GPU: steady state if it never leaves.
//
// End-to-end means: time from "input is in memory" to "the CPU has read the
// result back", i.e. exactly what one offloaded time step would cost.
//
// Build: nvcc -O3 -arch=native -Xcompiler -fopenmp unified_memory_bench.cu -o unified_memory_bench

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);std::exit(1);}}while(0)

__host__ __device__ inline uint64_t mix64(uint64_t z){
  z+=0x9E3779B97F4A7C15ULL; z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL;
  z=(z^(z>>27))*0x94D049BB133111EBULL; return z^(z>>31);
}
__host__ __device__ inline double cuni(uint64_t s,uint32_t i,uint32_t j,uint32_t st){
  uint64_t k=s; k=mix64(k^(uint64_t(i)*0x100000001B3ULL));
  k=mix64(k^(uint64_t(j)*0x100000001B3ULL)); k=mix64(k^(uint64_t(st)*0x100000001B3ULL));
  return (k>>11)*(1.0/9007199254740992.0);
}

// Propagate + per-pair stochastic decision, one block per cell.
__global__ void kern(double* x,double* y,double* z,const double* vx,const double* vy,
                     const double* vz,const int* cs,const int* cc,const int* cp,
                     const long* po,int nc,double xs,double dt,double Vc,uint64_t seed,
                     uint8_t* dec){
  for(int c=blockIdx.x;c<nc;c+=gridDim.x){
    int start=cs[c], cnt=cc[c]; long base=po[c];
    for(int a=threadIdx.x;a<cnt;a+=blockDim.x){ int g=cp[start+a]; x[g]+=vx[g]*dt; y[g]+=vy[g]*dt; z[g]+=vz[g]*dt; }
    __syncthreads();
    int np=cnt*(cnt-1)/2;
    for(int idx=threadIdx.x;idx<np;idx+=blockDim.x){
      int a=0,rem=idx; while(rem>=(cnt-1-a)){rem-=(cnt-1-a);a++;} int b=a+1+rem;
      int gi=cp[start+a], gj=cp[start+b];
      double dvx=vx[gi]-vx[gj],dvy=vy[gi]-vy[gj],dvz=vz[gi]-vz[gj];
      double vrel=sqrt(dvx*dvx+dvy*dvy+dvz*dvz);
      dec[base+idx]=(cuni(seed,gi,gj,1)<= xs*vrel*dt/Vc)?1:0;
    }
  }
}

double cpu_consume(const uint8_t* dec,long n){ long s=0;
#pragma omp parallel for reduction(+:s)
  for(long k=0;k<n;k++) s+=dec[k]; return double(s); }

double now(){
#ifdef _OPENMP
  return omp_get_wtime();
#else
  return double(clock())/CLOCKS_PER_SEC;
#endif
}

int main(int argc,char** argv){
  int per=argc>1?atoi(argv[1]):64, cpd=argc>2?atoi(argv[2]):24;
  int nc=cpd*cpd*cpd; double L=0.5,Vc=L*L*L,dt=0.1,xs=3.0; uint64_t seed=0xC0FFEE;
  // geometry
  std::vector<int> cs(nc),cc(nc),cp; std::vector<long> po(nc); long npairs=0; int gid=0;
  for(int c=0;c<nc;c++){cs[c]=gid;cc[c]=per;po[c]=npairs;npairs+=long(per)*(per-1)/2;
    for(int k=0;k<per;k++){cp.push_back(gid);gid++;}}
  long N=gid;
  printf("GB10 unified-memory benchmark: %ld particles, %d cells, %ld pairs\n\n",N,nc,npairs);
  int tpb=256, blocks=nc;
  int nthreads=1;
#ifdef _OPENMP
  nthreads=omp_get_max_threads();
#endif

  auto fill=[&](double*x,double*y,double*z,double*vx,double*vy,double*vz){
    for(long g=0;g<N;g++){ x[g]=cuni(seed,g,1,0); y[g]=cuni(seed,g,2,0); z[g]=cuni(seed,g,3,0);
      vx[g]=cuni(seed,g,4,0)-0.5; vy[g]=cuni(seed,g,5,0)-0.5; vz[g]=cuni(seed,g,6,0)-0.5; } };

  // ---------- CPU baseline ----------
  std::vector<double> hx(N),hy(N),hz(N),hvx(N),hvy(N),hvz(N); std::vector<uint8_t> hdec(npairs);
  fill(hx.data(),hy.data(),hz.data(),hvx.data(),hvy.data(),hvz.data());
  double t0=now();
#pragma omp parallel for schedule(dynamic)
  for(int c=0;c<nc;c++){ int start=cs[c],cnt=cc[c]; long base=po[c];
    for(int a=0;a<cnt;a++){int g=cp[start+a]; hx[g]+=hvx[g]*dt;hy[g]+=hvy[g]*dt;hz[g]+=hvz[g]*dt;}
    long kk=0; for(int a=0;a<cnt;a++)for(int b=a+1;b<cnt;b++,kk++){int gi=cp[start+a],gj=cp[start+b];
      double dvx=hvx[gi]-hvx[gj],dvy=hvy[gi]-hvy[gj],dvz=hvz[gi]-hvz[gj];
      double vrel=sqrt(dvx*dvx+dvy*dvy+dvz*dvz); hdec[base+kk]=(cuni(seed,gi,gj,1)<=xs*vrel*dt/Vc)?1:0; } }
  double cpu_sum=cpu_consume(hdec.data(),npairs);
  double t_cpu=now()-t0;
  printf("[CPU]  OpenMP %2d threads               : %7.2f ms   (collisions=%.0f)\n",nthreads,t_cpu*1e3,cpu_sum);

  // device geometry (shared, resident — geometry never changes per step)
  int *dcs,*dcc,*dcp; long* dpo;
  CK(cudaMalloc(&dcs,nc*sizeof(int))); CK(cudaMalloc(&dcc,nc*sizeof(int)));
  CK(cudaMalloc(&dcp,N*sizeof(int)));  CK(cudaMalloc(&dpo,nc*sizeof(long)));
  CK(cudaMemcpy(dcs,cs.data(),nc*sizeof(int),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dcc,cc.data(),nc*sizeof(int),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dcp,cp.data(),N*sizeof(int),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dpo,po.data(),nc*sizeof(long),cudaMemcpyHostToDevice));

  auto check=[&](double s,const char* tag){ if(fabs(s-cpu_sum)>0.5) printf("   !! %s mismatch %.0f vs %.0f\n",tag,s,cpu_sum); };

  // ---------- [explicit] cudaMalloc + memcpy ----------
  {
    double *x,*y,*z,*vx,*vy,*vz; uint8_t* dec;
    CK(cudaMalloc(&x,N*8));CK(cudaMalloc(&y,N*8));CK(cudaMalloc(&z,N*8));
    CK(cudaMalloc(&vx,N*8));CK(cudaMalloc(&vy,N*8));CK(cudaMalloc(&vz,N*8));CK(cudaMalloc(&dec,npairs));
    std::vector<uint8_t> out(npairs);
    // warmup
    kern<<<blocks,tpb>>>(x,y,z,vx,vy,vz,dcs,dcc,dcp,dpo,nc,xs,dt,Vc,seed,dec); CK(cudaDeviceSynchronize());
    double t=now();
    CK(cudaMemcpy(x,hx.data(),N*8,cudaMemcpyHostToDevice));CK(cudaMemcpy(y,hy.data(),N*8,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(z,hz.data(),N*8,cudaMemcpyHostToDevice));CK(cudaMemcpy(vx,hvx.data(),N*8,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(vy,hvy.data(),N*8,cudaMemcpyHostToDevice));CK(cudaMemcpy(vz,hvz.data(),N*8,cudaMemcpyHostToDevice));
    kern<<<blocks,tpb>>>(x,y,z,vx,vy,vz,dcs,dcc,dcp,dpo,nc,xs,dt,Vc,seed,dec);
    CK(cudaMemcpy(out.data(),dec,npairs,cudaMemcpyDeviceToHost)); // implies sync
    double s=cpu_consume(out.data(),npairs); double te=now()-t;
    printf("[GPU explicit] malloc+memcpy H2D/D2H   : %7.2f ms   speedup %.2fx\n",te*1e3,t_cpu/te); check(s,"explicit");
    cudaFree(x);cudaFree(y);cudaFree(z);cudaFree(vx);cudaFree(vy);cudaFree(vz);cudaFree(dec);
  }

  // ---------- [managed] on-demand page migration ----------
  {
    double *x,*y,*z,*vx,*vy,*vz; uint8_t* dec;
    CK(cudaMallocManaged(&x,N*8));CK(cudaMallocManaged(&y,N*8));CK(cudaMallocManaged(&z,N*8));
    CK(cudaMallocManaged(&vx,N*8));CK(cudaMallocManaged(&vy,N*8));CK(cudaMallocManaged(&vz,N*8));
    CK(cudaMallocManaged(&dec,npairs));
    // warmup
    fill(x,y,z,vx,vy,vz);
    kern<<<blocks,tpb>>>(x,y,z,vx,vy,vz,dcs,dcc,dcp,dpo,nc,xs,dt,Vc,seed,dec); CK(cudaDeviceSynchronize());
    fill(x,y,z,vx,vy,vz); // CPU re-produces input (now resident on CPU)
    double t=now();
    kern<<<blocks,tpb>>>(x,y,z,vx,vy,vz,dcs,dcc,dcp,dpo,nc,xs,dt,Vc,seed,dec);
    CK(cudaDeviceSynchronize());
    double s=cpu_consume(dec,npairs); double te=now()-t;
    printf("[GPU managed]  cudaMallocManaged        : %7.2f ms   speedup %.2fx\n",te*1e3,t_cpu/te); check(s,"managed");
    cudaFree(x);cudaFree(y);cudaFree(z);cudaFree(vx);cudaFree(vy);cudaFree(vz);cudaFree(dec);
  }

  // ---------- [ATS/malloc] plain host malloc straight into the kernel ----------
  {
    double *x=hx.data(),*y=hy.data(),*z=hz.data(),*vx=hvx.data(),*vy=hvy.data(),*vz=hvz.data();
    // fresh copies so propagation starts from the same input
    std::vector<double> X(hx),Y(hy),Z(hz); fill(X.data(),Y.data(),Z.data(),hvx.data(),hvy.data(),hvz.data());
    uint8_t* dec=(uint8_t*)malloc(npairs);
    // warmup (does ATS work at all?)
    cudaError_t we;
    kern<<<blocks,tpb>>>(X.data(),Y.data(),Z.data(),vx,vy,vz,dcs,dcc,dcp,dpo,nc,xs,dt,Vc,seed,dec);
    we=cudaDeviceSynchronize();
    if(we!=cudaSuccess){ printf("[GPU ATS/malloc] not supported (%s)\n",cudaGetErrorString(we)); }
    else {
      std::vector<double> X2(hx),Y2(hy),Z2(hz); fill(X2.data(),Y2.data(),Z2.data(),hvx.data(),hvy.data(),hvz.data());
      double t=now();
      kern<<<blocks,tpb>>>(X2.data(),Y2.data(),Z2.data(),vx,vy,vz,dcs,dcc,dcp,dpo,nc,xs,dt,Vc,seed,dec);
      CK(cudaDeviceSynchronize());
      double s=cpu_consume(dec,npairs); double te=now()-t;
      printf("[GPU ATS/malloc] malloc -> kernel      : %7.2f ms   speedup %.2fx\n",te*1e3,t_cpu/te); check(s,"ats");
    }
    free(dec);
  }

  // ---------- [resident] data already on device (steady state) ----------
  {
    double *x,*y,*z,*vx,*vy,*vz; uint8_t* dec;
    CK(cudaMalloc(&x,N*8));CK(cudaMalloc(&y,N*8));CK(cudaMalloc(&z,N*8));
    CK(cudaMalloc(&vx,N*8));CK(cudaMalloc(&vy,N*8));CK(cudaMalloc(&vz,N*8));CK(cudaMalloc(&dec,npairs));
    CK(cudaMemcpy(vx,hvx.data(),N*8,cudaMemcpyHostToDevice));CK(cudaMemcpy(vy,hvy.data(),N*8,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(vz,hvz.data(),N*8,cudaMemcpyHostToDevice));CK(cudaMemcpy(x,hx.data(),N*8,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(y,hy.data(),N*8,cudaMemcpyHostToDevice));CK(cudaMemcpy(z,hz.data(),N*8,cudaMemcpyHostToDevice));
    kern<<<blocks,tpb>>>(x,y,z,vx,vy,vz,dcs,dcc,dcp,dpo,nc,xs,dt,Vc,seed,dec); CK(cudaDeviceSynchronize());
    cudaEvent_t a,b; cudaEventCreate(&a);cudaEventCreate(&b);
    cudaEventRecord(a);
    kern<<<blocks,tpb>>>(x,y,z,vx,vy,vz,dcs,dcc,dcp,dpo,nc,xs,dt,Vc,seed,dec);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms,a,b);
    printf("[GPU resident] kernel only (no transfer): %7.2f ms   speedup %.2fx\n",ms,(t_cpu*1e3)/ms);
    cudaFree(x);cudaFree(y);cudaFree(z);cudaFree(vx);cudaFree(vy);cudaFree(vz);cudaFree(dec);
  }

  printf("\nNote: end-to-end = CPU input already in memory -> compute -> CPU reads result back.\n");
  cudaFree(dcs);cudaFree(dcc);cudaFree(dcp);cudaFree(dpo);
  return 0;
}
