#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include <bench/BenchTimer.h>




inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

// caffe2 im2col
void Im2col(
    const float* data_im,
    const int channels,
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int dilation_h,
    const int dilation_w,
    const int pad_t,
    const int pad_l,
    const int pad_b,
    const int pad_r,
    const int stride_h,
    const int stride_w,
    float* data_col) {
  const int output_h =
      (height + pad_b + pad_t - (dilation_h * (kernel_h - 1) + 1)) / stride_h +
      1;
  const int output_w =
      (width + pad_l + pad_r - (dilation_w * (kernel_w - 1) + 1)) / stride_w +
      1;


  // Fast path for zero padding and no dilation
  // From Torch, THNN_(unfolded_copy)
  if (dilation_h == 1 && dilation_w == 1 && pad_l == 0 && pad_r == 0 &&
      pad_t == 0 && pad_b == 0) {
    for (auto k = 0; k < channels * kernel_h * kernel_w; k++) {
      const auto nip = k / (kernel_h * kernel_w);
      const auto rest = k % (kernel_h * kernel_w);
      const auto kh = rest / kernel_w;
      const auto kw = rest % kernel_w;
      auto* dst = data_col + nip * (kernel_h * kernel_w * output_h * output_w) +
          kh * (kernel_w * output_h * output_w) + kw * (output_h * output_w);
      const auto* src = data_im + nip * (height * width);
      for (auto y = 0; y < output_h; y++) {
        const auto iy = y * stride_h + kh;
        const auto ix = kw;
        if (stride_w == 1) {
          memcpy(
              dst + (y * output_w),
              src + (iy * width + ix),
              sizeof(float) * output_w);
        } else {
          for (auto x = 0; x < output_w; x++) {
            memcpy(
                dst + (y * output_w + x),
                src + (iy * width + ix + x * stride_w),
                sizeof(float));
          }
        }
      }
    }
    return;
  }

  // Fast path for equal padding
  if (pad_l == pad_r && pad_t == pad_b) {
    // From Intel, https://github.com/BVLC/caffe/pull/3536
    const int pad_h = pad_t;
    const int pad_w = pad_l;
    const int channel_size = height * width;
    for (int channel = channels; channel--; data_im += channel_size) {
      for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
        for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
          int input_row = -pad_h + kernel_row * dilation_h;
          for (int output_rows = output_h; output_rows; output_rows--) {
            if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
              for (int output_cols = output_w; output_cols; output_cols--) {
                *(data_col++) = 0;
              }
            } else {
              int input_col = -pad_w + kernel_col * dilation_w;
              for (int output_col = output_w; output_col; output_col--) {
                if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                  *(data_col++) = data_im[input_row * width + input_col];
                } else {
                  *(data_col++) = 0;
                }
                input_col += stride_w;
              }
            }
            input_row += stride_h;
          }
        }
      }
    }
    return;
  }

  // Baseline
  const int dkernel_h = dilation_h * (kernel_h - 1) + 1;
  const int dkernel_w = dilation_w * (kernel_w - 1) + 1;

  int height_col = (height + pad_t + pad_b - dkernel_h) / stride_h + 1;
  int width_col = (width + pad_l + pad_r - dkernel_w) / stride_w + 1;

  int channels_col = channels * kernel_h * kernel_w;
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % kernel_w;
    int h_offset = (c / kernel_w) % kernel_h;
    int c_im = c / kernel_h / kernel_w;
    for (int h = 0; h < height_col; ++h) {
      for (int w = 0; w < width_col; ++w) {
        int h_pad = h * stride_h - pad_t + h_offset * dilation_h;
        int w_pad = w * stride_w - pad_l + w_offset * dilation_w;
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
          data_col[(c * height_col + h) * width_col + w] =
              data_im[(c_im * height + h_pad) * width + w_pad];
        else
          data_col[(c * height_col + h) * width_col + w] = 0;
      }
    }
  }

}

namespace Eigen {

namespace internal {

template<
  class T, 
  class Index>
struct conv_pack_rhs{
  void operator()(
    T* block, 
    const T* A, 
    Index C, 
    Index H, 
    Index W, 
    Index start_c, 
    Index start_i,
    Index start_j,
    Index lenC,
    Index lenH,
    Index lenW){


    if(start_c + lenC > C || start_i + lenH > H || start_j + lenW > W){
      std::cout << "conv pack rhs out of bound" << std::endl;
      exit(0);
    }

    Index channel_offset = H * W;
    const T* Aptr = A + start_c * channel_offset;

    Eigen::Map<Eigen::Matrix<T, -1, -1, Eigen::RowMajor>> blockmap(block, lenC * lenH, lenW);

    for(Index c = 0; c < lenC; ++c){
      Eigen::Map<const Eigen::Matrix<T, -1, -1, Eigen::RowMajor>> Amap(Aptr, H, W);
      blockmap.block(c * lenH, 0, lenH, lenW) = Amap.block(start_i, start_j, lenH, lenW);
      Aptr += channel_offset;
    }

  }
};

template<typename _LhsScalar, typename _RhsScalar, class Index>
class conv_level3_blocking
{
    typedef _LhsScalar LhsScalar;
    typedef _RhsScalar RhsScalar;

  protected:
    LhsScalar* m_blockA;
    RhsScalar* m_blockB;

    Index m_mc;
    Index m_nc;
    Index m_kc;

  public:

    conv_level3_blocking()
      : m_blockA(0), m_blockB(0), m_mc(0), m_nc(0), m_kc(0)
    {}

    inline Index mc() const { return m_mc; }
    inline Index nc() const { return m_nc; }
    inline Index kc() const { return m_kc; }

    inline LhsScalar* blockA() { return m_blockA; }
    inline RhsScalar* blockB() { return m_blockB; }
};

template<typename LhsScalar, typename RhsScalar, int KcFactor, typename Index>
void convEvaluateProductBlockingSizesHeuristic(Index& k, Index& m, Index& n, Index num_threads = 1)
{
  typedef gebp_traits<LhsScalar,RhsScalar> Traits;


  std::ptrdiff_t l1, l2, l3;
  manage_caching_sizes(GetAction, &l1, &l2, &l3);

  if (num_threads > 1) {
    typedef typename Traits::ResScalar ResScalar;
    enum {
      kdiv = KcFactor * (Traits::mr * sizeof(LhsScalar) + Traits::nr * sizeof(RhsScalar)),
      ksub = Traits::mr * Traits::nr * sizeof(ResScalar),
      kr = 8,
      mr = Traits::mr,
      nr = Traits::nr
    };

    const Index k_cache = (numext::mini<Index>)((l1-ksub)/kdiv, 320);
    if (k_cache < k) {
      k = k_cache - (k_cache % kr);
      eigen_internal_assert(k > 0);
    }

    const Index n_cache = (l2-l1) / (nr * sizeof(RhsScalar) * k);
    const Index n_per_thread = numext::div_ceil(n, num_threads);
    if (n_cache <= n_per_thread) {
      eigen_internal_assert(n_cache >= static_cast<Index>(nr));
      n = n_cache - (n_cache % nr);
      eigen_internal_assert(n > 0);
    } else {
      n = (numext::mini<Index>)(n, (n_per_thread + nr - 1) - ((n_per_thread + nr - 1) % nr));
    }

    if (l3 > l2) {
      const Index m_cache = (l3-l2) / (sizeof(LhsScalar) * k * num_threads);
      const Index m_per_thread = numext::div_ceil(m, num_threads);
      if(m_cache < m_per_thread && m_cache >= static_cast<Index>(mr)) {
        m = m_cache - (m_cache % mr);
        eigen_internal_assert(m > 0);
      } else {
        m = (numext::mini<Index>)(m, (m_per_thread + mr - 1) - ((m_per_thread + mr - 1) % mr));
      }
    }
  }
  else {
#ifdef EIGEN_DEBUG_SMALL_PRODUCT_BLOCKS
    l1 = 9*1024;
    l2 = 32*1024;
    l3 = 512*1024;
#endif


    if((numext::maxi)(k,(numext::maxi)(m,n))<48)
      return;

    typedef typename Traits::ResScalar ResScalar;
    enum {
      k_peeling = 8,
      k_div = KcFactor * (Traits::mr * sizeof(LhsScalar) + Traits::nr * sizeof(RhsScalar)),
      k_sub = Traits::mr * Traits::nr * sizeof(ResScalar)
    };


    const Index max_kc = numext::maxi<Index>(((l1-k_sub)/k_div) & (~(k_peeling-1)),1);
    const Index old_k = k;
    if(k>max_kc)
    {

      k = (k%max_kc)==0 ? max_kc
                        : max_kc - k_peeling * ((max_kc-1-(k%max_kc))/(k_peeling*(k/max_kc+1)));

      eigen_internal_assert(((old_k/k) == (old_k/max_kc)) && "the number of sweeps has to remain the same");
    }


    #ifdef EIGEN_DEBUG_SMALL_PRODUCT_BLOCKS
    const Index actual_l2 = l3;
    #else
    const Index actual_l2 = 1572864; // == 1.5 MB
    #endif

 
    Index max_nc;
    const Index lhs_bytes = m * k * sizeof(LhsScalar);
    const Index remaining_l1 = l1- k_sub - lhs_bytes;

    max_nc = (3*actual_l2)/(2*2*max_kc*sizeof(RhsScalar));
    Index nc = numext::mini<Index>(actual_l2/(2*k*sizeof(RhsScalar)), max_nc) & (~(Traits::nr-1));
    if(n>nc)
    {
      n = (n%nc)==0 ? nc
                    : (nc - Traits::nr * ((nc/*-1*/-(n%nc))/(Traits::nr*(n/nc+1))));
    }
    else if(old_k==k)
    {
      Index problem_size = k*n*sizeof(LhsScalar);
      Index actual_lm = actual_l2;
      Index max_mc = m;
      if(problem_size<=1024)
      {
        actual_lm = l1;
      }
      else if(l3!=0 && problem_size<=32768)
      {
        actual_lm = l2;
        max_mc = (numext::mini<Index>)(576,max_mc);
      }
      Index mc = (numext::mini<Index>)(actual_lm/(3*k*sizeof(LhsScalar)), max_mc);
      if (mc > Traits::mr) mc -= mc % Traits::mr;
      else if (mc==0) return;
      m = (m%mc)==0 ? mc
                    : (mc - Traits::mr * ((mc/*-1*/-(m%mc))/(Traits::mr*(m/mc+1))));
    }
  }
}

template<typename LhsScalar, typename RhsScalar, int KcFactor, typename Index>
void convComputeProductBlockingSizes(Index& k, Index& m, Index& n, Index num_threads = 1)
{
  if (!useSpecificBlockingSizes(k, m, n)) {
    convEvaluateProductBlockingSizesHeuristic<LhsScalar, RhsScalar, KcFactor, Index>(k, m, n, num_threads);
  }
}

template<int StorageOrder, typename LhsScalar, typename RhsScalar, int MaxRows, int MaxCols, int MaxDepth, class Index> 
class conv_blocking_space;

template<int StorageOrder, typename _LhsScalar, typename _RhsScalar, int MaxRows, int MaxCols, int MaxDepth, class Index>
class conv_blocking_space
  : public conv_level3_blocking<
      typename conditional<StorageOrder==RowMajor,_RhsScalar,_LhsScalar>::type,
      typename conditional<StorageOrder==RowMajor,_LhsScalar,_RhsScalar>::type,
      Index>
{
    enum {
      Transpose = StorageOrder==RowMajor
    };
    typedef typename conditional<Transpose,_RhsScalar,_LhsScalar>::type LhsScalar;
    typedef typename conditional<Transpose,_LhsScalar,_RhsScalar>::type RhsScalar;
    typedef gebp_traits<LhsScalar,RhsScalar> Traits;

    Index m_sizeA;
    Index m_sizeB;
    Index conv_mc;
    Index conv_channelsc;
    Index conv_ohc;
    Index conv_owc;

  public:

    Index mc(){return conv_mc;}
    Index channelsc(){return conv_channelsc;}
    Index ohc(){return conv_ohc;}
    Index owc(){return conv_owc;}


    conv_blocking_space(Index m, Index channels, Index oh, Index ow, Index num_threads = 1, bool l3_blocking = true)
    {
      Index rows = m;
      Index cols = oh * ow;
      Index depth = 9 * channels;
      this->m_mc = Transpose ? cols : rows;
      this->m_nc = Transpose ? rows : cols;
      this->m_kc = depth;

      if(l3_blocking)
      {
        convComputeProductBlockingSizes<LhsScalar,RhsScalar,1>(this->m_kc, this->m_mc, this->m_nc, num_threads);
      }
      else  
      {
        Index n = this->m_nc;
        convComputeProductBlockingSizes<LhsScalar,RhsScalar,1>(this->m_kc, this->m_mc, n, num_threads);
      }

      m_sizeA = this->m_mc * this->m_kc;
      m_sizeB = this->m_kc * this->m_nc;

      Index mc =  this->m_mc;
      Index nc =  this->m_nc;
      Index kc =  this->m_kc;

      conv_mc = mc;
      conv_channelsc = kc / 9;
      Index remain = kc * nc / conv_channelsc;
      Index sqrtremain = static_cast<Index>(std::sqrt(static_cast<double>(remain)));
      conv_owc = std::min(ow,std::max(static_cast<Index>(8), sqrtremain / 8 * 8));
      conv_ohc = std::max(static_cast<Index>(1),std::min(oh, remain / (conv_owc + 2) - 2));
    }

    void initParallel(Index rows, Index cols, Index depth, Index num_threads)
    {
      this->m_mc = Transpose ? cols : rows;
      this->m_nc = Transpose ? rows : cols;
      this->m_kc = depth;

      eigen_internal_assert(this->m_blockA==0 && this->m_blockB==0);
      Index m = this->m_mc;
      convComputeProductBlockingSizes<LhsScalar,RhsScalar,1>(this->m_kc, m, this->m_nc, num_threads);
      m_sizeA = this->m_mc * this->m_kc;
      m_sizeB = this->m_kc * this->m_nc;
    }

    void allocateA()
    {
      if(this->m_blockA==0)
        this->m_blockA = aligned_new<LhsScalar>(m_sizeA);
    }

    void allocateB()
    {
      if(this->m_blockB==0)
        this->m_blockB = aligned_new<RhsScalar>(m_sizeB);
    }

    void allocateAll()
    {
      allocateA();
      allocateB();
    }

    ~conv_blocking_space()
    {
      aligned_delete(this->m_blockA, m_sizeA);
      aligned_delete(this->m_blockB, m_sizeB);
    }
};

template<typename _LhsScalar, typename _RhsScalar, bool _ConjLhs, bool _ConjRhs>
class half_gebp_traits
{
public:
  typedef _LhsScalar LhsScalar;
  typedef _RhsScalar RhsScalar;
  typedef typename ScalarBinaryOpTraits<LhsScalar, RhsScalar>::ReturnType ResScalar;

  enum {
    ConjLhs = _ConjLhs,
    ConjRhs = _ConjRhs,
    Vectorizable = packet_traits<LhsScalar>::Vectorizable && packet_traits<RhsScalar>::Vectorizable,
    LhsPacketSize = Vectorizable ? packet_traits<LhsScalar>::size / 2 : 1,
    RhsPacketSize = Vectorizable ? packet_traits<RhsScalar>::size / 2 : 1,
    ResPacketSize = Vectorizable ? packet_traits<ResScalar>::size / 2 : 1,
    
    NumberOfRegisters = EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS,

    nr = 4,

    default_mr = (EIGEN_PLAIN_ENUM_MIN(16,NumberOfRegisters)/2/nr)*LhsPacketSize,
#if defined(EIGEN_HAS_SINGLE_INSTRUCTION_MADD) && !defined(EIGEN_VECTORIZE_ALTIVEC) && !defined(EIGEN_VECTORIZE_VSX)
    mr = Vectorizable ? 3*LhsPacketSize : default_mr,
#else
    mr = default_mr,
#endif
    
    LhsProgress = LhsPacketSize,
    RhsProgress = 1
  };

  typedef typename packet_traits<LhsScalar>::half  _LhsPacket;
  typedef typename packet_traits<RhsScalar>::half  _RhsPacket;
  typedef typename packet_traits<ResScalar>::half  _ResPacket;

  typedef typename conditional<Vectorizable,_LhsPacket,LhsScalar>::type LhsPacket;
  typedef typename conditional<Vectorizable,_RhsPacket,RhsScalar>::type RhsPacket;
  typedef typename conditional<Vectorizable,_ResPacket,ResScalar>::type ResPacket;

  typedef ResPacket AccPacket;
  
  EIGEN_STRONG_INLINE void initAcc(AccPacket& p)
  {
    p = pset1<ResPacket>(ResScalar(0));
  }
  
  EIGEN_STRONG_INLINE void broadcastRhs(const RhsScalar* b, RhsPacket& b0, RhsPacket& b1, RhsPacket& b2, RhsPacket& b3)
  {
    pbroadcast4(b, b0, b1, b2, b3);
  }
  
  
  template<typename RhsPacketType>
  EIGEN_STRONG_INLINE void loadRhs(const RhsScalar* b, RhsPacketType& dest) const
  {
    dest = pset1<RhsPacketType>(*b);
  }
  
  EIGEN_STRONG_INLINE void loadRhsQuad(const RhsScalar* b, RhsPacket& dest) const
  {
    dest = ploadquad<RhsPacket>(b);
  }

  template<typename LhsPacketType>
  EIGEN_STRONG_INLINE void loadLhs(const LhsScalar* a, LhsPacketType& dest) const
  {
    dest = pload<LhsPacketType>(a);
  }

  template<typename LhsPacketType>
  EIGEN_STRONG_INLINE void loadLhsUnaligned(const LhsScalar* a, LhsPacketType& dest) const
  {
    dest = ploadu<LhsPacketType>(a);
  }

  template<typename LhsPacketType, typename RhsPacketType, typename AccPacketType>
  EIGEN_STRONG_INLINE void madd(const LhsPacketType& a, const RhsPacketType& b, AccPacketType& c, AccPacketType& tmp) const
  {
    conj_helper<LhsPacketType,RhsPacketType,ConjLhs,ConjRhs> cj;
#ifdef EIGEN_HAS_SINGLE_INSTRUCTION_MADD
    EIGEN_UNUSED_VARIABLE(tmp);
    c = cj.pmadd(a,b,c);
#else
    tmp = b; tmp = cj.pmul(a,tmp); c = padd(c,tmp);
#endif
  }

  EIGEN_STRONG_INLINE void acc(const AccPacket& c, const ResPacket& alpha, ResPacket& r) const
  {
    r = pmadd(c,alpha,r);
  }
  
  template<typename ResPacketHalf>
  EIGEN_STRONG_INLINE void acc(const ResPacketHalf& c, const ResPacketHalf& alpha, ResPacketHalf& r) const
  {
    r = pmadd(c,alpha,r);
  }

};

#ifdef EIGEN_HAS_SINGLE_INSTRUCTION_CJMADD
  #define CJMADD(CJ,A,B,C,T)  C = CJ.pmadd(A,B,C);
#else
  #define CJMADD(CJ,A,B,C,T)  gebp_madd(CJ,A,B,C,T);
#endif

template<typename LhsScalar, typename RhsScalar, typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
struct new_conv_kernel
{
  typedef gebp_traits<LhsScalar,RhsScalar,ConjugateLhs,ConjugateRhs> Traits;
  typedef typename Traits::ResScalar ResScalar;
  typedef typename Traits::LhsPacket LhsPacket;
  typedef typename Traits::RhsPacket RhsPacket;
  typedef typename Traits::ResPacket ResPacket;
  typedef typename Traits::AccPacket AccPacket;

  typedef gebp_traits<RhsScalar,LhsScalar,ConjugateRhs,ConjugateLhs> _SwappedTraits;
  typedef half_gebp_traits<RhsScalar,LhsScalar,ConjugateRhs,ConjugateLhs> _HalfSwappedTraits;
  typedef typename std::conditional<
    _SwappedTraits::LhsProgress >= 8,
    _HalfSwappedTraits,
    _SwappedTraits>::type SwappedTraits;

  typedef typename SwappedTraits::ResScalar SResScalar;
  typedef typename SwappedTraits::LhsPacket SLhsPacket;
  typedef typename SwappedTraits::RhsPacket SRhsPacket;
  typedef typename SwappedTraits::ResPacket SResPacket;
  typedef typename SwappedTraits::AccPacket SAccPacket;

  typedef typename DataMapper::LinearMapper LinearMapper;

  enum {
    Vectorizable  = Traits::Vectorizable,
    LhsProgress   = Traits::LhsProgress,
    RhsProgress   = Traits::RhsProgress,
    ResPacketSize = Traits::ResPacketSize
  };

  EIGEN_DONT_INLINE
  void operator()(const DataMapper& res, const LhsScalar* blockA, const RhsScalar* blockB,
                  Index rows, Index depth, Index oh, Index ow, Index ow_stride,
                  ResScalar alpha, Index strideA=-1, Index strideB=-1);
};


template<typename LhsScalar, typename RhsScalar, typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
EIGEN_DONT_INLINE
void new_conv_kernel<LhsScalar,RhsScalar,Index,DataMapper,mr,nr,ConjugateLhs,ConjugateRhs>
  ::operator()(const DataMapper& res, const LhsScalar* blockA, const RhsScalar* blockB,
               Index rows, Index depth, Index oh, Index ow, Index ow_stride,
               ResScalar alpha, Index strideA, Index strideB)
{
  Traits traits;
  SwappedTraits straits;

  // conv 3x3
  if(depth % 9 != 0){
    std::cout << "depth must be multiple of 9" << std::endl;
    exit(0);
  }


  Index cols = oh * ow;
  if(strideA==-1) strideA = depth;
  if(strideB==-1) strideB = depth;
  conj_helper<LhsScalar,RhsScalar,ConjugateLhs,ConjugateRhs> cj;
  Index packet_cols4 = nr>=4 ? (cols/4) * 4 : 0;
  const Index peeled_mc3 = mr>=3*Traits::LhsProgress ? (rows/(3*LhsProgress))*(3*LhsProgress) : 0;
  const Index peeled_mc2 = mr>=2*Traits::LhsProgress ? peeled_mc3+((rows-peeled_mc3)/(2*LhsProgress))*(2*LhsProgress) : 0;
  const Index peeled_mc1 = mr>=1*Traits::LhsProgress ? (rows/(1*LhsProgress))*(1*LhsProgress) : 0;
  enum { pk = 9 }; 
  const Index peeled_kc  = depth  / pk * pk;
  const Index prefetch_res_offset = 32/sizeof(ResScalar);    


  constexpr Index h_increase = 1;
  constexpr Index w_increase = 4;
  Index packet_ow4 = nr>=4 ? (ow/4) * 4 : 0;

  if(mr>=3*Traits::LhsProgress)
  {
    const Index l1 = defaultL1CacheSize; 
    /*to be changed*/const Index actual_panel_rows = (3*LhsProgress) * std::max<Index>(1,( (l1 - sizeof(ResScalar)*mr*nr - depth*nr*sizeof(RhsScalar)) / (depth * sizeof(LhsScalar) * 3*LhsProgress) ));
    for(Index i1=0; i1<peeled_mc3; i1+=actual_panel_rows)
    {
      const Index actual_panel_end = (std::min)(i1+actual_panel_rows, peeled_mc3);
      for(Index jh = 0; jh < oh; jh += h_increase)
      {
        const Index actual_jh_end = (std::min)(jh + h_increase, oh);
        Index vert = jh;

        for(Index jw = 0; jw < packet_ow4; jw += w_increase)
        {
          const Index actual_jw_end = (std::min)(jw + w_increase, ow);
          Index hori = jw;

          for(Index i=i1; i<actual_panel_end; i+=3*LhsProgress)
          {
            // for(Index vert = jh; vert < actual_jh_end; ++vert)
            // for(Index hori = jw; hori < actual_jw_end; hori += nr)
            {
              const LhsScalar* blA = &blockA[i*strideA];
              prefetch(&blA[0]);

              AccPacket C0, C1, C2,  C3,
                        C4, C5, C6,  C7,
                        C8, C9, C10, C11;
              traits.initAcc(C0);  traits.initAcc(C1);  traits.initAcc(C2);  traits.initAcc(C3);
              traits.initAcc(C4);  traits.initAcc(C5);  traits.initAcc(C6);  traits.initAcc(C7);
              traits.initAcc(C8);  traits.initAcc(C9);  traits.initAcc(C10); traits.initAcc(C11);

              LinearMapper r0 = res.getLinearMapper(i, vert * ow_stride + hori + 0);
              LinearMapper r1 = res.getLinearMapper(i, vert * ow_stride + hori + 1);
              LinearMapper r2 = res.getLinearMapper(i, vert * ow_stride + hori + 2);
              LinearMapper r3 = res.getLinearMapper(i, vert * ow_stride + hori + 3);


              r0.prefetch(0);
              r1.prefetch(0);
              r2.prefetch(0);
              r3.prefetch(0);

              const RhsScalar* blB = &blockB[vert * (ow + 2) + hori];
              prefetch(&blB[0]);
              LhsPacket A0, A1;

              for(Index k=0; k<peeled_kc; k+=pk)
              {
                EIGEN_ASM_COMMENT("begin gebp micro kernel 3pX4");
                RhsPacket B_0, T0;
                LhsPacket A2;

      #define EIGEN_GEBP_ONESTEP(K) \
                do { \
                  EIGEN_ASM_COMMENT("begin step of gebp micro kernel 3pX4"); \
                  EIGEN_ASM_COMMENT("Note: these asm comments work around bug 935!"); \
                  internal::prefetch(blA+(3*K+16)*LhsProgress); \
                  if (EIGEN_ARCH_ARM) { internal::prefetch(blB+(4*K+16)*RhsProgress); } /* Bug 953 */ \
                  traits.loadLhs(&blA[(0+3*K)*LhsProgress], A0);  \
                  traits.loadLhs(&blA[(1+3*K)*LhsProgress], A1);  \
                  traits.loadLhs(&blA[(2+3*K)*LhsProgress], A2);  \
                  traits.loadRhs(blB + (0+(K / 3) * (ow + 2) + K % 3)*Traits::RhsProgress, B_0); \
                  traits.madd(A0, B_0, C0, T0); \
                  traits.madd(A1, B_0, C4, T0); \
                  traits.madd(A2, B_0, C8, B_0); \
                  traits.loadRhs(blB + (1+(K / 3) * (ow + 2) + K % 3)*Traits::RhsProgress, B_0); \
                  traits.madd(A0, B_0, C1, T0); \
                  traits.madd(A1, B_0, C5, T0); \
                  traits.madd(A2, B_0, C9, B_0); \
                  traits.loadRhs(blB + (2+(K / 3) * (ow + 2) + K % 3)*Traits::RhsProgress, B_0); \
                  traits.madd(A0, B_0, C2,  T0); \
                  traits.madd(A1, B_0, C6,  T0); \
                  traits.madd(A2, B_0, C10, B_0); \
                  traits.loadRhs(blB + (3+(K / 3) * (ow + 2) + K % 3)*Traits::RhsProgress, B_0); \
                  traits.madd(A0, B_0, C3 , T0); \
                  traits.madd(A1, B_0, C7,  T0); \
                  traits.madd(A2, B_0, C11, B_0); \
                  EIGEN_ASM_COMMENT("end step of gebp micro kernel 3pX4"); \
                } while(false)

                internal::prefetch(blB);
                EIGEN_GEBP_ONESTEP(0);
                EIGEN_GEBP_ONESTEP(1);
                EIGEN_GEBP_ONESTEP(2);
                EIGEN_GEBP_ONESTEP(3);
                EIGEN_GEBP_ONESTEP(4);
                EIGEN_GEBP_ONESTEP(5);
                EIGEN_GEBP_ONESTEP(6);
                EIGEN_GEBP_ONESTEP(7);
                EIGEN_GEBP_ONESTEP(8);

                blB += (ow + 2) * (oh + 2);
                blA += pk*3*Traits::LhsProgress;

                EIGEN_ASM_COMMENT("end gebp micro kernel 3pX4");
              }


      #undef EIGEN_GEBP_ONESTEP

              ResPacket R0, R1, R2;
              ResPacket alphav = pset1<ResPacket>(alpha);

              R0 = r0.loadPacket(0 * Traits::ResPacketSize);
              R1 = r0.loadPacket(1 * Traits::ResPacketSize);
              R2 = r0.loadPacket(2 * Traits::ResPacketSize);
              traits.acc(C0, alphav, R0);
              traits.acc(C4, alphav, R1);
              traits.acc(C8, alphav, R2);
              r0.storePacket(0 * Traits::ResPacketSize, R0);
              r0.storePacket(1 * Traits::ResPacketSize, R1);
              r0.storePacket(2 * Traits::ResPacketSize, R2);

              R0 = r1.loadPacket(0 * Traits::ResPacketSize);
              R1 = r1.loadPacket(1 * Traits::ResPacketSize);
              R2 = r1.loadPacket(2 * Traits::ResPacketSize);
              traits.acc(C1, alphav, R0);
              traits.acc(C5, alphav, R1);
              traits.acc(C9, alphav, R2);
              r1.storePacket(0 * Traits::ResPacketSize, R0);
              r1.storePacket(1 * Traits::ResPacketSize, R1);
              r1.storePacket(2 * Traits::ResPacketSize, R2);

              R0 = r2.loadPacket(0 * Traits::ResPacketSize);
              R1 = r2.loadPacket(1 * Traits::ResPacketSize);
              R2 = r2.loadPacket(2 * Traits::ResPacketSize);
              traits.acc(C2, alphav, R0);
              traits.acc(C6, alphav, R1);
              traits.acc(C10, alphav, R2);
              r2.storePacket(0 * Traits::ResPacketSize, R0);
              r2.storePacket(1 * Traits::ResPacketSize, R1);
              r2.storePacket(2 * Traits::ResPacketSize, R2);

              R0 = r3.loadPacket(0 * Traits::ResPacketSize);
              R1 = r3.loadPacket(1 * Traits::ResPacketSize);
              R2 = r3.loadPacket(2 * Traits::ResPacketSize);
              traits.acc(C3, alphav, R0);
              traits.acc(C7, alphav, R1);
              traits.acc(C11, alphav, R2);
              r3.storePacket(0 * Traits::ResPacketSize, R0);
              r3.storePacket(1 * Traits::ResPacketSize, R1);
              r3.storePacket(2 * Traits::ResPacketSize, R2);          
            
            }
          }
        }
      }
      
      if(packet_ow4 != ow)
      for(Index vert = 0; vert < oh; ++vert)
      for(Index hori=packet_ow4; hori<ow; ++hori)
      {
        {
          for(Index i=i1; i<actual_panel_end; i+=3*LhsProgress)
          {
          const LhsScalar* blA = &blockA[i*strideA];
          prefetch(&blA[0]);

          AccPacket C0, C4, C8;
          traits.initAcc(C0);
          traits.initAcc(C4);
          traits.initAcc(C8);

          LinearMapper r0 = res.getLinearMapper(i, vert * ow_stride + hori + 0);

          r0.prefetch(0);

          const RhsScalar* blB = &blockB[vert * (ow + 2) + hori];

          LhsPacket A0, A1, A2;
          
          for(Index k=0; k<peeled_kc; k+=pk)
          {
            EIGEN_ASM_COMMENT("begin gebp micro kernel 3pX1");
            RhsPacket B_0;
  #define EIGEN_GEBGP_ONESTEP(K) \
            do { \
              EIGEN_ASM_COMMENT("begin step of gebp micro kernel 3pX1"); \
              EIGEN_ASM_COMMENT("Note: these asm comments work around bug 935!"); \
              traits.loadLhs(&blA[(0+3*K)*LhsProgress], A0);  \
              traits.loadLhs(&blA[(1+3*K)*LhsProgress], A1);  \
              traits.loadLhs(&blA[(2+3*K)*LhsProgress], A2);  \
              traits.loadRhs(blB + (0+(K / 3) * (ow + 2) + K % 3)*Traits::RhsProgress, B_0); \
              traits.madd(A0, B_0, C0, B_0); \
              traits.madd(A1, B_0, C4, B_0); \
              traits.madd(A2, B_0, C8, B_0); \
              EIGEN_ASM_COMMENT("end step of gebp micro kernel 3pX1"); \
            } while(false)
        
            EIGEN_GEBGP_ONESTEP(0);
            EIGEN_GEBGP_ONESTEP(1);
            EIGEN_GEBGP_ONESTEP(2);
            EIGEN_GEBGP_ONESTEP(3);
            EIGEN_GEBGP_ONESTEP(4);
            EIGEN_GEBGP_ONESTEP(5);
            EIGEN_GEBGP_ONESTEP(6);
            EIGEN_GEBGP_ONESTEP(7);
            EIGEN_GEBGP_ONESTEP(8);

            blB += (ow + 2) * (oh + 2);
            blA += pk*3*Traits::LhsProgress;

            EIGEN_ASM_COMMENT("end gebp micro kernel 3pX1");
          }


  #undef EIGEN_GEBGP_ONESTEP
          ResPacket R0, R1, R2;
          ResPacket alphav = pset1<ResPacket>(alpha);

          R0 = r0.loadPacket(0 * Traits::ResPacketSize);
          R1 = r0.loadPacket(1 * Traits::ResPacketSize);
          R2 = r0.loadPacket(2 * Traits::ResPacketSize);
          traits.acc(C0, alphav, R0);
          traits.acc(C4, alphav, R1);
          traits.acc(C8, alphav, R2);
          r0.storePacket(0 * Traits::ResPacketSize, R0);
          r0.storePacket(1 * Traits::ResPacketSize, R1);
          r0.storePacket(2 * Traits::ResPacketSize, R2);          
          }
        }
      }



    }
  }





  if(mr>=2*Traits::LhsProgress)
  {
    const Index l1 = defaultL1CacheSize; 
    Index actual_panel_rows = (2*LhsProgress) * std::max<Index>(1,( (l1 - sizeof(ResScalar)*mr*nr - depth*nr*sizeof(RhsScalar)) / (depth * sizeof(LhsScalar) * 2*LhsProgress) ));
    const Index actual_panel_cols = 12 * nr;

    for(Index i1=peeled_mc3; i1<peeled_mc2; i1+=actual_panel_rows)
    {
      Index actual_panel_end = (std::min)(i1+actual_panel_rows, peeled_mc2);
      for(Index jh = 0; jh < oh; jh += h_increase)
      for(Index jw = 0; jw < packet_ow4; jw += w_increase)
      {

        const Index actual_jh_end = (std::min)(jh + h_increase, oh);
        const Index actual_jw_end = (std::min)(jw + w_increase, ow);

        for(Index i=i1; i<actual_panel_end; i+=2*LhsProgress)
        {
          for(Index vert = jh; vert < actual_jh_end; ++vert)
          for(Index hori = jw; hori < actual_jw_end; hori += nr)
          {
            const LhsScalar* blA = &blockA[i*strideA];
            prefetch(&blA[0]);

            AccPacket C0, C1, C2, C3,
                      C4, C5, C6, C7;
            traits.initAcc(C0); traits.initAcc(C1); traits.initAcc(C2); traits.initAcc(C3);
            traits.initAcc(C4); traits.initAcc(C5); traits.initAcc(C6); traits.initAcc(C7);

            LinearMapper r0 = res.getLinearMapper(i, vert * ow_stride + hori + 0);
            LinearMapper r1 = res.getLinearMapper(i, vert * ow_stride + hori + 1);
            LinearMapper r2 = res.getLinearMapper(i, vert * ow_stride + hori + 2);
            LinearMapper r3 = res.getLinearMapper(i, vert * ow_stride + hori + 3);


            r0.prefetch(prefetch_res_offset);
            r1.prefetch(prefetch_res_offset);
            r2.prefetch(prefetch_res_offset);
            r3.prefetch(prefetch_res_offset);

            const RhsScalar* blB = &blockB[vert * (ow + 2) + hori];
            prefetch(&blB[0]);
            LhsPacket A0, A1;

            for(Index k=0; k<peeled_kc; k+=pk)
            {
              EIGEN_ASM_COMMENT("begin gebp micro kernel 2pX4");
              RhsPacket B_0, B1, B2, B3, T0;

     #define EIGEN_GEBGP_ONESTEP(K) \
              do {                                                                \
                EIGEN_ASM_COMMENT("begin step of gebp micro kernel 2pX4");        \
                EIGEN_ASM_COMMENT("Note: these asm comments work around bug 935!"); \
                traits.loadLhs(&blA[(0+2*K)*LhsProgress], A0);                    \
                traits.loadLhs(&blA[(1+2*K)*LhsProgress], A1);                    \
                traits.broadcastRhs(blB + (0+(K / 3) * (ow + 2) + K % 3)*Traits::RhsProgress, B_0, B1, B2, B3);  \
                traits.madd(A0, B_0, C0, T0);                                     \
                traits.madd(A1, B_0, C4, B_0);                                    \
                traits.madd(A0, B1,  C1, T0);                                     \
                traits.madd(A1, B1,  C5, B1);                                     \
                traits.madd(A0, B2,  C2, T0);                                     \
                traits.madd(A1, B2,  C6, B2);                                     \
                traits.madd(A0, B3,  C3, T0);                                     \
                traits.madd(A1, B3,  C7, B3);                                     \
                EIGEN_ASM_COMMENT("end step of gebp micro kernel 2pX4");          \
              } while(false)
              
              internal::prefetch(blB+(48+0));
              EIGEN_GEBGP_ONESTEP(0);
              EIGEN_GEBGP_ONESTEP(1);
              EIGEN_GEBGP_ONESTEP(2);
              EIGEN_GEBGP_ONESTEP(3);
              internal::prefetch(blB+(48+16));
              EIGEN_GEBGP_ONESTEP(4);
              EIGEN_GEBGP_ONESTEP(5);
              EIGEN_GEBGP_ONESTEP(6);
              EIGEN_GEBGP_ONESTEP(7);
              EIGEN_GEBGP_ONESTEP(8);

              blB += (ow + 2) * (oh + 2);
              blA += pk*(2*Traits::LhsProgress);

              EIGEN_ASM_COMMENT("end gebp micro kernel 2pX4");
            }
 
    #undef EIGEN_GEBGP_ONESTEP

            ResPacket R0, R1, R2, R3;
            ResPacket alphav = pset1<ResPacket>(alpha);

            R0 = r0.loadPacket(0 * Traits::ResPacketSize);
            R1 = r0.loadPacket(1 * Traits::ResPacketSize);
            R2 = r1.loadPacket(0 * Traits::ResPacketSize);
            R3 = r1.loadPacket(1 * Traits::ResPacketSize);
            traits.acc(C0, alphav, R0);
            traits.acc(C4, alphav, R1);
            traits.acc(C1, alphav, R2);
            traits.acc(C5, alphav, R3);
            r0.storePacket(0 * Traits::ResPacketSize, R0);
            r0.storePacket(1 * Traits::ResPacketSize, R1);
            r1.storePacket(0 * Traits::ResPacketSize, R2);
            r1.storePacket(1 * Traits::ResPacketSize, R3);

            R0 = r2.loadPacket(0 * Traits::ResPacketSize);
            R1 = r2.loadPacket(1 * Traits::ResPacketSize);
            R2 = r3.loadPacket(0 * Traits::ResPacketSize);
            R3 = r3.loadPacket(1 * Traits::ResPacketSize);
            traits.acc(C2,  alphav, R0);
            traits.acc(C6,  alphav, R1);
            traits.acc(C3,  alphav, R2);
            traits.acc(C7,  alphav, R3);
            r2.storePacket(0 * Traits::ResPacketSize, R0);
            r2.storePacket(1 * Traits::ResPacketSize, R1);
            r3.storePacket(0 * Traits::ResPacketSize, R2);
            r3.storePacket(1 * Traits::ResPacketSize, R3);
          
          }
        }
      }






      for(Index hori=packet_ow4; hori<ow; ++hori)
      for(Index vert = 0; vert < oh; ++vert)
      {
        for(Index i=i1; i<actual_panel_end; i+=2*LhsProgress)
        {
          const LhsScalar* blA = &blockA[i*strideA];
          prefetch(&blA[0]);

          AccPacket C0, C4;
          traits.initAcc(C0);
          traits.initAcc(C4);

          LinearMapper r0 = res.getLinearMapper(i, vert * ow_stride + hori + 0);
          r0.prefetch(prefetch_res_offset);

          const RhsScalar* blB = &blockB[vert * (ow + 2) + hori];

          LhsPacket A0, A1;

          for(Index k=0; k<peeled_kc; k+=pk)
          {
            EIGEN_ASM_COMMENT("begin gebp micro kernel 2pX1");
            RhsPacket B_0, B1;
        
    #define EIGEN_GEBGP_ONESTEP(K) \
            do {                                                                  \
              EIGEN_ASM_COMMENT("begin step of gebp micro kernel 2pX1");          \
              EIGEN_ASM_COMMENT("Note: these asm comments work around bug 935!"); \
              traits.loadLhs(&blA[(0+2*K)*LhsProgress], A0);                      \
              traits.loadLhs(&blA[(1+2*K)*LhsProgress], A1);                      \
              traits.loadRhs(blB + (0+(K / 3) * (ow + 2) + K % 3)*Traits::RhsProgress, B_0); \
              traits.madd(A0, B_0, C0, B1);                                       \
              traits.madd(A1, B_0, C4, B_0);                                      \
              EIGEN_ASM_COMMENT("end step of gebp micro kernel 2pX1");            \
            } while(false)
        
            EIGEN_GEBGP_ONESTEP(0);
            EIGEN_GEBGP_ONESTEP(1);
            EIGEN_GEBGP_ONESTEP(2);
            EIGEN_GEBGP_ONESTEP(3);
            EIGEN_GEBGP_ONESTEP(4);
            EIGEN_GEBGP_ONESTEP(5);
            EIGEN_GEBGP_ONESTEP(6);
            EIGEN_GEBGP_ONESTEP(7);
            EIGEN_GEBGP_ONESTEP(8);

            blB += (ow + 2) * (oh + 2);
            blA += pk*2*Traits::LhsProgress;

            EIGEN_ASM_COMMENT("end gebp micro kernel 2pX1");
          }

    #undef EIGEN_GEBGP_ONESTEP
          ResPacket R0, R1;
          ResPacket alphav = pset1<ResPacket>(alpha);

          R0 = r0.loadPacket(0 * Traits::ResPacketSize);
          R1 = r0.loadPacket(1 * Traits::ResPacketSize);
          traits.acc(C0, alphav, R0);
          traits.acc(C4, alphav, R1);
          r0.storePacket(0 * Traits::ResPacketSize, R0);
          r0.storePacket(1 * Traits::ResPacketSize, R1);
        }
      }
    }
  }

  // bound: operation / overhead or L1: wo gu ji de
  if(mr>=1*Traits::LhsProgress)
  {
    const Index actual_panel_cols = 1 * nr;

    for(Index i=peeled_mc2; i<peeled_mc1; i+=1*LhsProgress)
    {
      for(Index jh = 0; jh < oh; jh += h_increase)
      for(Index jw = 0; jw < packet_ow4; jw += w_increase)
      // for(Index jw = 0; jw < ow; jw += w_increase)
      {
        const Index actual_jh_end = (std::min)(jh + h_increase, oh);
        const Index actual_jw_end = (std::min)(jw + w_increase, ow);
        for(Index vert = jh; vert < actual_jh_end; ++vert)
        for(Index hori = jw; hori < actual_jw_end; hori += nr)
        {
          const LhsScalar* blA = &blockA[i*strideA];
          prefetch(&blA[0]);

          AccPacket C0, C1, C2, C3;
          traits.initAcc(C0);
          traits.initAcc(C1);
          traits.initAcc(C2);
          traits.initAcc(C3);

          LinearMapper r0 = res.getLinearMapper(i, vert * ow_stride + hori + 0);
          LinearMapper r1 = res.getLinearMapper(i, vert * ow_stride + hori + 1);
          LinearMapper r2 = res.getLinearMapper(i, vert * ow_stride + hori + 2);
          LinearMapper r3 = res.getLinearMapper(i, vert * ow_stride + hori + 3);


          r0.prefetch(prefetch_res_offset);
          r1.prefetch(prefetch_res_offset);
          r2.prefetch(prefetch_res_offset);
          r3.prefetch(prefetch_res_offset);

          const RhsScalar* blB = &blockB[vert * (ow + 2) + hori];
          prefetch(&blB[0]);
          LhsPacket A0;

          for(Index k=0; k<peeled_kc; k+=pk)
          {
            EIGEN_ASM_COMMENT("begin gebp micro kernel 1pX4");
            // RhsPacket B_0, B1, B2, B3;
            RhsPacket B_0, B1, B2, B3, B4, B5;
               
  // #define EIGEN_GEBGP_ONESTEP(K) \
  //           do {                                                                \
  //             EIGEN_ASM_COMMENT("begin step of gebp micro kernel 1pX4");        \
  //             EIGEN_ASM_COMMENT("Note: these asm comments work around bug 935!"); \
  //             traits.loadLhs(&blA[(0+1*K)*LhsProgress], A0);                    \
  //             traits.broadcastRhs(blB + (0+(K / 3) * (ow + 2) + K % 3)*Traits::RhsProgress, B_0, B1, B2, B3);  \
  //             traits.madd(A0, B_0, C0, B_0);                                    \
  //             traits.madd(A0, B1,  C1, B1);                                     \
  //             traits.madd(A0, B2,  C2, B2);                                     \
  //             traits.madd(A0, B3,  C3, B3);                                     \
  //             EIGEN_ASM_COMMENT("end step of gebp micro kernel 1pX4");          \
  //           } while(false)


  #define EIGEN_GEBGP_THREESTEP(K) \
            do {                                                                \
              EIGEN_ASM_COMMENT("begin step of gebp micro kernel 1pX4");        \
              EIGEN_ASM_COMMENT("Note: these asm comments work around bug 935!"); \
              traits.loadLhs(&blA[(0+3*K)*LhsProgress], A0);                    \
              traits.broadcastRhs(blB + (0+K*(ow + 2))*Traits::RhsProgress, B_0, B1, B2, B3);  \
              traits.madd(A0, B_0, C0, B_0);                                    \
              traits.madd(A0, B1,  C1, B1);                                     \
              traits.madd(A0, B2,  C2, B2);                                     \
              traits.madd(A0, B3,  C3, B3);                                     \
              traits.loadLhs(&blA[(1+3*K)*LhsProgress], A0);                    \
              traits.loadRhs(blB + (0+K*(ow + 2) + 4)*Traits::RhsProgress, B4); \
              traits.madd(A0, B1,  C0, B1);                                    \
              traits.madd(A0, B2,  C1, B2);                                     \
              traits.madd(A0, B3,  C2, B3);                                     \
              traits.madd(A0, B4,  C3, B4);                                     \
              traits.loadLhs(&blA[(2+3*K)*LhsProgress], A0);                    \
              traits.loadRhs(blB + (0+K*(ow + 2) + 5)*Traits::RhsProgress, B5); \
              traits.madd(A0, B2,  C0, B2);                                    \
              traits.madd(A0, B3,  C1, B3);                                     \
              traits.madd(A0, B4,  C2, B4);                                     \
              traits.madd(A0, B5,  C3, B5);                                     \
              EIGEN_ASM_COMMENT("end step of gebp micro kernel 1pX4");          \
            } while(false)

            
            internal::prefetch(blB+(48+0));
            EIGEN_GEBGP_THREESTEP(0);
            // EIGEN_GEBGP_ONESTEP(0);
            // EIGEN_GEBGP_ONESTEP(1);
            // EIGEN_GEBGP_ONESTEP(2);
            // EIGEN_GEBGP_ONESTEP(3);
            internal::prefetch(blB+(48+16));
            EIGEN_GEBGP_THREESTEP(1);
            EIGEN_GEBGP_THREESTEP(2);
            // EIGEN_GEBGP_ONESTEP(4);
            // EIGEN_GEBGP_ONESTEP(5);
            // EIGEN_GEBGP_ONESTEP(6);
            // EIGEN_GEBGP_ONESTEP(7);
            // EIGEN_GEBGP_ONESTEP(8);

            blB += (ow + 2) * (oh + 2);
            blA += pk*1*LhsProgress;

            EIGEN_ASM_COMMENT("end gebp micro kernel 1pX4");
          }

  #undef EIGEN_GEBGP_ONESTEP

          ResPacket R0, R1;
          ResPacket alphav = pset1<ResPacket>(alpha);

          R0 = r0.loadPacket(0 * Traits::ResPacketSize);
          R1 = r1.loadPacket(0 * Traits::ResPacketSize);
          traits.acc(C0, alphav, R0);
          traits.acc(C1,  alphav, R1);
          r0.storePacket(0 * Traits::ResPacketSize, R0);
          r1.storePacket(0 * Traits::ResPacketSize, R1);

          R0 = r2.loadPacket(0 * Traits::ResPacketSize);
          R1 = r3.loadPacket(0 * Traits::ResPacketSize);
          traits.acc(C2,  alphav, R0);
          traits.acc(C3,  alphav, R1);
          r2.storePacket(0 * Traits::ResPacketSize, R0);
          r3.storePacket(0 * Traits::ResPacketSize, R1);
        
        }
      }

      for(Index hori=packet_ow4; hori<ow; ++hori)
      for(Index vert = 0; vert < oh; ++vert)
      {
        const LhsScalar* blA = &blockA[i*strideA];
        prefetch(&blA[0]);

        AccPacket C0;
        traits.initAcc(C0);

        LinearMapper r0 = res.getLinearMapper(i, vert * ow_stride + hori + 0);

        const RhsScalar* blB = &blockB[vert * (ow + 2) + hori];
        LhsPacket A0;

        for(Index k=0; k<peeled_kc; k+=pk)
        {
          EIGEN_ASM_COMMENT("begin gebp micro kernel 1pX1");
          RhsPacket B_0;
      
#define EIGEN_GEBGP_ONESTEP(K) \
          do {                                                                \
            EIGEN_ASM_COMMENT("begin step of gebp micro kernel 1pX1");        \
            EIGEN_ASM_COMMENT("Note: these asm comments work around bug 935!"); \
            traits.loadLhs(&blA[(0+1*K)*LhsProgress], A0);                    \
            traits.loadRhs(blB + (0+(K / 3) * (ow + 2) + K % 3)*Traits::RhsProgress, B_0); \
            traits.madd(A0, B_0, C0, B_0);                                    \
            EIGEN_ASM_COMMENT("end step of gebp micro kernel 1pX1");          \
          } while(false);

          EIGEN_GEBGP_ONESTEP(0);
          EIGEN_GEBGP_ONESTEP(1);
          EIGEN_GEBGP_ONESTEP(2);
          EIGEN_GEBGP_ONESTEP(3);
          EIGEN_GEBGP_ONESTEP(4);
          EIGEN_GEBGP_ONESTEP(5);
          EIGEN_GEBGP_ONESTEP(6);
          EIGEN_GEBGP_ONESTEP(7);
          EIGEN_GEBGP_ONESTEP(8);

          blB += (ow + 2) * (oh + 2);
          blA += pk*1*Traits::LhsProgress;

          EIGEN_ASM_COMMENT("end gebp micro kernel 1pX1");
        }


#undef EIGEN_GEBGP_ONESTEP
        ResPacket R0;
        ResPacket alphav = pset1<ResPacket>(alpha);
        R0 = r0.loadPacket(0 * Traits::ResPacketSize);
        traits.acc(C0, alphav, R0);
        r0.storePacket(0 * Traits::ResPacketSize, R0);
      }
    }
  }

  // operation or L1
  if(peeled_mc1<rows)
  {

    typedef typename conditional<SwappedTraits::LhsProgress>=8,typename unpacket_traits<SResPacket>::half,SResPacket>::type SResPacketHalf;
    typedef typename conditional<SwappedTraits::LhsProgress>=8,typename unpacket_traits<SLhsPacket>::half,SLhsPacket>::type SLhsPacketHalf;
    typedef typename conditional<SwappedTraits::LhsProgress>=8,typename unpacket_traits<SLhsPacket>::half,SRhsPacket>::type SRhsPacketHalf;
    typedef typename conditional<SwappedTraits::LhsProgress>=8,typename unpacket_traits<SAccPacket>::half,SAccPacket>::type SAccPacketHalf;

    for(Index vert = 0; vert < oh; ++vert)
    for(Index hori = 0; hori < packet_ow4; hori += nr)
    // for(Index hori = 0; hori < ow; hori += nr)
    // for(Index j2=0; j2<packet_cols4; j2+=nr)
    {
      // Index vert = j2 / ow;
      // Index hori = j2 % ow;

      for(Index i=peeled_mc1; i<rows; i+=1)
      {

        const LhsScalar* blA = &blockA[i*strideA];
        prefetch(&blA[0]);
        const RhsScalar* blB = &blockB[vert * (ow + 2) + hori];



        typedef typename unpacket_traits<SResPacket>::half SResPacketHalf;
        if ((SwappedTraits::LhsProgress % 4) == 0 &&
            (SwappedTraits::LhsProgress <= 8) &&
            (SwappedTraits::LhsProgress!=8 || unpacket_traits<SResPacketHalf>::size==nr))
        {

          SAccPacket C0, C1, C2, C3, C4, C5, C6, C7, C8;

          straits.initAcc(C0);
          straits.initAcc(C1);
          straits.initAcc(C2);
          straits.initAcc(C3);
          straits.initAcc(C4);
          straits.initAcc(C5);
          straits.initAcc(C6);
          straits.initAcc(C7);
          straits.initAcc(C8);


          const Index spk   = 1;
          const Index endk  = (depth/spk)*spk;
          const Index endk4 = (depth/(spk*9))*(spk*9);

          Index k=0;
          for(; k<endk4; k+=9*spk)
          {
            SLhsPacket A0, A1, A2;
            SRhsPacket B_0, B_1, B_2;

            straits.loadLhsUnaligned(blB + (0+(0 / 3) * (ow + 2) + 0 % 3), A0);
            straits.loadLhsUnaligned(blB + (0+(1 / 3) * (ow + 2) + 1 % 3), A1);
            straits.loadLhsUnaligned(blB + (0+(2 / 3) * (ow + 2) + 2 % 3), A2);
            straits.loadRhsQuad(blA+0*spk, B_0);
            straits.loadRhsQuad(blA+1*spk, B_1);
            straits.loadRhsQuad(blA+2*spk, B_2);
            straits.madd(A0,B_0,C0,B_0);
            straits.madd(A1,B_1,C1,B_1);
            straits.madd(A2,B_2,C2,B_2);

            straits.loadLhsUnaligned(blB + (0+(3 / 3) * (ow + 2) + 3 % 3), A0);
            straits.loadLhsUnaligned(blB + (0+(4 / 3) * (ow + 2) + 4 % 3), A1);
            straits.loadLhsUnaligned(blB + (0+(5 / 3) * (ow + 2) + 5 % 3), A2);
            straits.loadRhsQuad(blA+3*spk, B_0);
            straits.loadRhsQuad(blA+4*spk, B_1);
            straits.loadRhsQuad(blA+5*spk, B_2);
            straits.madd(A0,B_0,C3,B_0);
            straits.madd(A1,B_1,C4,B_1);
            straits.madd(A2,B_2,C5,B_2);

            straits.loadLhsUnaligned(blB + (0+(6 / 3) * (ow + 2) + 6 % 3), A0);
            straits.loadLhsUnaligned(blB + (0+(7 / 3) * (ow + 2) + 7 % 3), A1);
            straits.loadLhsUnaligned(blB + (0+(8 / 3) * (ow + 2) + 8 % 3), A2);
            straits.loadRhsQuad(blA+6*spk, B_0);
            straits.loadRhsQuad(blA+7*spk, B_1);
            straits.loadRhsQuad(blA+8*spk, B_2);
            straits.madd(A0,B_0,C6,B_0);
            straits.madd(A1,B_1,C7,B_1);
            straits.madd(A2,B_2,C8,B_2);

            blB += (ow + 2) * (oh + 2);
            blA += 9*spk;
          }
          C0 = padd(padd(padd(padd(C0,C1),padd(C2,C3)), padd(padd(C4,C5),padd(C6,C7))), C8);

          SResPacket R = res.template gatherPacket<SResPacket>(i, vert * ow_stride + hori + 0);
          SResPacket alphav = pset1<SResPacket>(alpha);
          straits.acc(C0, alphav, R);
          res.scatterPacket(i, vert * ow_stride + hori + 0, R);
        }
        else 
        {
          std::cout << "not implemented yet" << std::endl;
          exit(0);
          ResScalar C0(0), C1(0), C2(0), C3(0);

          // for(Index k=0; k<depth; k++)
          // {
          //   LhsScalar A0;
          //   RhsScalar B_0, B_1;

          //   A0 = blA[k];

          //   B_0 = blB[0];
          //   B_1 = blB[1];
          //   CJMADD(cj,A0,B_0,C0,  B_0);
          //   CJMADD(cj,A0,B_1,C1,  B_1);
            
          //   B_0 = blB[2];
          //   B_1 = blB[3];
          //   CJMADD(cj,A0,B_0,C2,  B_0);
          //   CJMADD(cj,A0,B_1,C3,  B_1);
            
          //   blB += 4;
          // }
          // res(i, j2 + 0) += alpha * C0;
          // res(i, j2 + 1) += alpha * C1;
          // res(i, j2 + 2) += alpha * C2;
          // res(i, j2 + 3) += alpha * C3;
        }
      }
    }


    //bound operation
    for(Index hori=packet_ow4; hori<ow; ++hori)
    for(Index vert = 0; vert < oh; ++vert)
    // for(Index j2=packet_cols4; j2<cols; j2++)
    {
      // std::cout << "this not implement yet" << std::endl;
      // exit(0);
      for(Index i=peeled_mc1; i<rows; i+=1)
      {
        const LhsScalar* blA = &blockA[i*strideA];
        prefetch(&blA[0]);
        ResScalar C0(0);
        const RhsScalar* blB = &blockB[vert * (ow + 2) + hori];
        // const RhsScalar* blB = &blockB[j2*strideB];
        for(Index k=0; k<depth; k+=pk)
        // for(Index k=0; k<depth; k++)
        {
          LhsScalar A0 = blA[k + 0];
          RhsScalar B_0 = blB[(0+(0 / 3) * (ow + 2) + 0 % 3)*Traits::RhsProgress];
          CJMADD(cj, A0, B_0, C0, B_0);
          A0 = blA[k + 1];
          B_0 = blB[(0+(1 / 3) * (ow + 2) + 1 % 3)*Traits::RhsProgress];
          CJMADD(cj, A0, B_0, C0, B_0);
          A0 = blA[k + 2];
          B_0 = blB[(0+(2 / 3) * (ow + 2) + 2 % 3)*Traits::RhsProgress];
          CJMADD(cj, A0, B_0, C0, B_0);
          A0 = blA[k + 3];
          B_0 = blB[(0+(3 / 3) * (ow + 2) + 3 % 3)*Traits::RhsProgress];
          CJMADD(cj, A0, B_0, C0, B_0);
          A0 = blA[k + 4];
          B_0 = blB[(0+(4 / 3) * (ow + 2) + 4 % 3)*Traits::RhsProgress];
          CJMADD(cj, A0, B_0, C0, B_0);
          A0 = blA[k + 5];
          B_0 = blB[(0+(5 / 3) * (ow + 2) + 5 % 3)*Traits::RhsProgress];
          CJMADD(cj, A0, B_0, C0, B_0);
          A0 = blA[k + 6];
          B_0 = blB[(0+(6 / 3) * (ow + 2) + 6 % 3)*Traits::RhsProgress];
          CJMADD(cj, A0, B_0, C0, B_0);
          A0 = blA[k + 7];
          B_0 = blB[(0+(7 / 3) * (ow + 2) + 7 % 3)*Traits::RhsProgress];
          CJMADD(cj, A0, B_0, C0, B_0);
          A0 = blA[k + 8];
          B_0 = blB[(0+(8 / 3) * (ow + 2) + 8 % 3)*Traits::RhsProgress];
          CJMADD(cj, A0, B_0, C0, B_0);

          blB += (ow + 2) * (oh + 2);

        }
        // res(i, j2) += alpha * C0;
        res(i, vert * ow_stride + hori + 0) += alpha * C0;
      }
    }
  }

}

#undef CJMADD

template<class T, class Index>
void conv3x3(Index m, Index channels, Index oh, Index ow,
  const T* _lhs, Index lhsStride,
  const T* _rhs,
  T* _res, Index resStride,
  T alpha,
  conv_blocking_space<Eigen::ColMajor, T, T, Eigen::Dynamic,Eigen::Dynamic,Eigen::Dynamic, size_t> blocking)
{

  using Traits = gebp_traits<T, T>;
  constexpr int LhsStorageOrder = Eigen::ColMajor;
  typedef const_blas_data_mapper<T, Index, LhsStorageOrder> LhsMapper;
  typedef blas_data_mapper<typename Traits::ResScalar, Index, Eigen::ColMajor> ResMapper;
  LhsMapper lhs(_lhs,lhsStride);
  ResMapper res(_res, resStride);

  Index rows = m;
  Index cols = oh * ow;
  Index depth = 9 * channels;


  Index channelsc = blocking.channelsc();
  Index ohc = blocking.ohc();
  Index owc = blocking.owc();

  Index kc = channelsc * 9;
  Index mc = (std::min)(rows, blocking.mc());
  Index nc = (std::min)(cols, ohc * owc);



  gemm_pack_lhs<T, Index, LhsMapper, Traits::mr, Traits::LhsProgress, LhsStorageOrder> pack_lhs;

  conv_pack_rhs<
    T, 
    Index> pack_rhs;

  new_conv_kernel<
    T,
    T,
    Index,
    ResMapper,
    Traits::mr,
    Traits::nr,
    0,
    0> conv_kernel;

  {

    std::size_t sizeA = kc*mc;
    std::size_t sizeB = kc*nc;
    ei_declare_aligned_stack_constructed_variable(T, blockA, sizeA, blocking.blockA());
    ei_declare_aligned_stack_constructed_variable(T, blockB, sizeB, blocking.blockB());


    const bool pack_rhs_once = mc!=rows && kc==depth && ohc==oh && owc == ow;

    for(Index i2=0; i2<rows; i2+=mc)
    {
      const Index actual_mc = (std::min)(i2+mc,rows)-i2;

      for(Index k2=0; k2<depth; k2+=kc)
      {
        const Index actual_kc = (std::min)(k2+kc,depth)-k2;

        pack_lhs(blockA, lhs.getSubMapper(i2,k2), actual_kc, actual_mc);

        for(Index u = 0; u < oh; u += ohc){
          const Index actual_ohc = (std::min)(u+ohc, oh) - u;
          for(Index v = 0; v < ow; v += owc){
            const Index actual_owc = (std::min)(v+owc,ow) - v;
            if((!pack_rhs_once) || (u==0 && v == 0))
            pack_rhs(
                blockB, 
                _rhs, 
                channels, 
                oh + 2, 
                ow + 2, 
                k2 / 9, 
                u,
                v,
                actual_kc / 9,
                actual_ohc + 2,
                actual_owc + 2);
            conv_kernel(
              res.getSubMapper(i2, u * ow + v), 
              blockA, 
              blockB, 
              actual_mc, 
              actual_kc, 
              actual_ohc, 
              actual_owc, 
              ow, 
              1.0);
          }
        }
      }
    }
  }

  
}

}
}




void benchmark(){

  constexpr size_t test_case_num = 8;

  // out_channel : in_channel : out_height : out_width
  size_t test_case[test_case_num][4] = {
    {8, 4, 32, 32},
    {8, 16, 32, 32},
    {8, 16, 64, 64},
    {16, 16, 64, 64},
    {16, 16, 256, 256},
    {24, 32, 256, 256},
    {48, 64, 256, 256},
    {64, 64, 512, 512}
  };

  for(int i = 0; i < test_case_num; ++i){
    
    size_t out_channel = test_case[i][0];
    size_t in_channel = test_case[i][1];
    size_t out_height = test_case[i][2];
    size_t out_width = test_case[i][3];

    size_t channel = in_channel;
    size_t oh = out_height;
    size_t ow = out_width;
    size_t H = out_height + 2;
    size_t W = out_width + 2;

    size_t m = out_channel;
    size_t n = out_height * out_width;
    size_t k = 9 * channel;

    std::cout << "test case: " << i << std::endl;
    std::cout << 
      "out_channel: " << out_channel << 
      " in_channel: " << in_channel << 
      " out_height: " << out_height << 
      " out_width: " << out_width << 
      std::endl;


    Eigen::Matrix<float, -1, -1, Eigen::ColMajor> 
      A1 = Eigen::Matrix<float, -1, -1, Eigen::ColMajor>::Random(m, k);
    Eigen::Matrix<float, -1, -1, Eigen::ColMajor> 
      A2(m, k);

    Eigen::Matrix<float, -1, -1, Eigen::RowMajor> 
      B = Eigen::Matrix<float, -1, -1, Eigen::RowMajor>::Random(channel * H, W);
    Eigen::Matrix<float, -1, -1, Eigen::RowMajor> 
      col = Eigen::Matrix<float, -1, -1, Eigen::RowMajor>(9 * channel, oh * ow);

    A2 = A1;


    Im2col(
      B.data(),
      channel,
      H,
      W,
      3,
      3,
      1,
      1,
      0,
      0,
      0,
      0,
      1,
      1,
      col.data());


    Eigen::Matrix<float, -1, -1, Eigen::ColMajor> C1 = A1 * col;
    Eigen::Matrix<float, -1, -1, Eigen::ColMajor> C2(m, n);

    Eigen::internal::conv_blocking_space<Eigen::ColMajor, float, float,Eigen::Dynamic,Eigen::Dynamic,Eigen::Dynamic, size_t> conv_blocking(m, channel, oh, ow);

    Eigen::internal::conv3x3( m,  channel,  oh,  ow, A2.data(),  m,
      B.data(),
      C2.data(),  m,
      1.0f,
      conv_blocking);

    std::cout << "error:" << (C1 - C2).norm() / (m * n) << std::endl;



    int tries = 4;
    int rep = std::max<int>(1,100000000/m/n/k);
    uint64_t op_count = rep * m * n * k;
    double GFLOP = (double)op_count / 1e9;
    Eigen::BenchTimer t1, t2;

    BENCH(t1, tries, rep, C1.noalias() = A1 * col);
    BENCH(t2, tries, rep, Eigen::internal::conv3x3( m,  channel,  oh,  ow, A2.data(),  m,
      B.data(),
      C2.data(),  m,
      1.0f,
      conv_blocking));



    std::cout << "GFLOPS by Eigen is: " << GFLOP / t1.best() << "\n";
    std::cout << "GFLOPS by direct-conv  is: " << GFLOP / t2.best() << "\n\n";
  }
}






int main(int argc, char* argv[]) {
  benchmark();
  return 0;
}