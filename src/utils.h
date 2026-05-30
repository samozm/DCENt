#define STRICT_R_HEADERS
#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS
#include <Rcpp.h>
#include <RcppEigen.h>
#include <vector>
#include <algorithm>
#include <random>
#include <typeinfo>
#include <deque>
#include <limits>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif
// [[Rcpp::depends(RcppEigen)]]

std::vector<std::vector<int>> MAP_to_idxs(const Eigen::MatrixXi & MAP,
                                          int n, int k, int t);

std::vector<std::vector<int>> MAP_to_idxs_2k(const Eigen::MatrixXi & MAP, int n, int k, int t);

double var(const Eigen::VectorXd & vec);

std::string printdims(const Eigen::MatrixXd & obj);

std::string printvec(const std::vector<double> & obj);

std::string BoolToString(bool b);

int BoolToInt(bool b);

int a_in_b(double a, const Eigen::VectorXd & b);

Eigen::ArrayXi loc_a_in_b(double a, const Eigen::VectorXd & b);

void build_V_list_from_master(std::vector<Eigen::MatrixXd> & V, const Eigen::MatrixXd & masterV, const std::vector<std::vector<int>> & map_idxs, int n, int k, int t);

void V_assemble_IP(const Eigen::Ref<const Eigen::MatrixXd> & masterV, 
                   Eigen::MatrixXd & V_out,
                   const std::vector<int> & map_idxs, int kt);

Eigen::MatrixXd covCalc(const Eigen::MatrixXd & X);

Eigen::MatrixXd covCalc(const Eigen::MatrixXd & X, const Eigen::MatrixXi & MAP);

Eigen::MatrixXd covCalc(const Eigen::MatrixXd & X, const std::vector<std::vector<int>> & map_idxs, int p);

void covCalc(const Eigen::MatrixXd & X, const std::vector<std::vector<int>> & map_idxs, 
             Eigen::MatrixXd & cov, int p);

void get_cov_stats(const Eigen::Ref<const Eigen::MatrixXd>& R, 
                   const std::vector<std::vector<int>>& active_lists, 
                   const std::vector<int>& indices,
                   Eigen::MatrixXd& SumXY, Eigen::MatrixXd& N, 
                   Eigen::MatrixXd& SumX_shared, Eigen::MatrixXd& SumRsq);

void build_cov_and_theta(const Eigen::MatrixXd& SumXY, const Eigen::MatrixXd& N, 
                         const Eigen::MatrixXd& SumX_shared, const Eigen::MatrixXd& SumRsq,
                         Eigen::MatrixXd& cov, Eigen::ArrayXXd& theta);

void vec2list(const std::vector<Eigen::MatrixXd>& vec, Rcpp::List & out);

void list2vec(std::vector<Eigen::MatrixXd>& vec, const Rcpp::List & list);

void find_all(const std::vector<int> & vec, const int & val, std::vector<int> & out_val, std::vector<int> & out_not_val);

void find_all(const Eigen::VectorXi & vec, const int & val, std::vector<int> & out_val, std::vector<int> & out_not_val);

Eigen::MatrixXd Et_assemble(const Eigen::VectorXd & E, 
                            const Eigen::MatrixXi & MAP, 
                            int i, int k, int t, int kt);

void Et_assemble_IP(const Eigen::Ref<const Eigen::VectorXd> & E, 
                    Eigen::VectorXd & Et,
                    const std::vector<int> & map_idxs, 
                    int k, int t);

Eigen::MatrixXd Z_assemble(const Eigen::MatrixXd & masterZ, 
                           const Eigen::MatrixXi & MAP,
                           int i, int k, int t, int kt);
void Z_assemble_IP(const Eigen::Ref<const Eigen::MatrixXd> & masterZt, 
                   Eigen::MatrixXd & Zt_out,
                   const std::vector<int> & map_idxs,
                   int k);

void Et_Z_assemble_IP(const Eigen::Ref<const Eigen::VectorXd> & masterE, 
                   Eigen::VectorXd & Et_out,
                   const Eigen::Ref<const Eigen::MatrixXd> & masterZt, 
                   Eigen::MatrixXd & Zt_out,
                   const Eigen::Ref<const Eigen::MatrixXi> & MAP,
                   int i, int k, int t, int kt);
                           void calc_ZDZ_plus_E_list(const Eigen::MatrixXd & masterZ,
                          const Eigen::MatrixXd & D, const Eigen::VectorXd & E,
                          std::vector<Eigen::MatrixXd> & out, 
                          const Eigen::MatrixXi & MAP,
                          int n, int k, int t);

Rcpp::List calc_ZDZ_plus_E_list(const Eigen::MatrixXd & masterZt,
                          const Eigen::MatrixXd & D, const Eigen::VectorXd & E,
                          const Eigen::MatrixXi & MAP,
                          int n, int k, int t);

void calc_ZDZ_plus_E(const Eigen::MatrixXd & masterZt,
                    const Eigen::MatrixXd & D, 
                    const Eigen::VectorXd & E,
                    Eigen::MatrixXd & out, 
                    const Eigen::MatrixXi & MAP,
                    int n, int k, int t, int nkt);

Rcpp::List calc_ZDZ_plus_E(const Eigen::MatrixXd & masterZt,
                          const Eigen::MatrixXd & D,
                          const Eigen::VectorXd & E,
                          const Eigen::MatrixXi & MAP,
                          int n, int k, int t, int nkt);

void estimate_beta(const Eigen::MatrixXd & X, const Eigen::VectorXd & y, 
                   const Eigen::VectorXi kt_vec, const std::vector<std::vector<int>> & map_idxs,
                   const Eigen::MatrixXd & masterV, Eigen::VectorXd & beta,
                   int n, int k, int t, bool verbose, double eigen_threshold=1e-5);

void estimate_beta2(const Eigen::Ref<const Eigen::MatrixXd> & X, 
                    const Eigen::Ref<const Eigen::VectorXd> & y, 
                    const Eigen::Ref<const Eigen::MatrixXd> & Zt,
                    const Eigen::Ref<const Eigen::MatrixXd> & D,
                    const Eigen::Ref<const Eigen::VectorXd> & E,
                    const std::vector<std::vector<int>> & map_idxs_kt,
                    Eigen::VectorXd & beta,
                    int n, int k, int t);

Eigen::VectorXd R_expand(const Eigen::VectorXd & R,
                         const Eigen::MatrixXi & MAP,
                         int idx, int q);

Eigen::MatrixXd RtR(const Eigen::MatrixXd & R, const std::vector<std::vector<int>>& active_lists);

void get_bounds(const Eigen::MatrixXd& cov, const Eigen::ArrayXXd& theta, double& lower, double& upper);

bool check_openmp();