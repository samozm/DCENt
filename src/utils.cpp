#define STRICT_R_HEADERS
#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS
#include "utils.h"
// [[Rcpp::depends(RcppEigen)]]


std::vector<std::vector<int>> MAP_to_idxs(const Eigen::MatrixXi & MAP,
                                          int n, int k, int t)
{
    std::vector<std::vector<int>> map_idxs(n);
    
    // Precompute the active indices for every subject exactly once
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < k*t; ++j) {
            if(MAP(i, j) == 1) {
                map_idxs[i].push_back(j);
            }
        }
    }
    return(map_idxs);
}

std::vector<std::vector<int>> MAP_to_idxs_2k(const Eigen::MatrixXi & MAP, int n, int k, int t)
{
    std::vector<std::vector<int>> map_idxs_2k(n);

    for(int i = 0; i < n; ++i) {
        // Random Effects Covariance Mapping (size 2k)
        for(int c = 0; c < 2*k; ++c) {
            int node = c / 2;
            bool has_data = false;
            for(int time_idx = 0; time_idx < t; ++time_idx) {
                if(MAP(i, node * t + time_idx) != 0) {
                    has_data = true; break;
                }
            }
            if(has_data) map_idxs_2k[i].push_back(c);
        }
    }
    return(map_idxs_2k);
}

double var(const Eigen::VectorXd & vec)
{
    return((vec.array() - vec.array().mean()).square().mean());
}

std::string printdims(const Eigen::MatrixXd & obj)
{
    return ("("+ std::to_string(obj.rows()) + ", " + std::to_string(obj.cols()) + ")\n");
}

std::string printvec(const std::vector<double> & obj)
{
  std::string out = std::to_string(obj[0]);
  for(int i=1; i<obj.size(); ++i)
  {
    out.append(", ").append(std::to_string(obj[i]));
  }
  return(out);
}

std::string BoolToString(bool b)
{
  return b ? "true" : "false";
}

int BoolToInt(bool b)
{
    return b ? 1 : 0;
}

int a_in_b(double a, const Eigen::VectorXd & b)
{
    for(double i: b)
    {
        if(std::abs(a-i)<0.01*b.cwiseAbs().maxCoeff())  //std::max(std::abs(b)))
        {
            return(1);
        }
    }
    return(0);
}

Eigen::ArrayXi loc_a_in_b(double a, const Eigen::VectorXd & b)
{
    std::deque<int> out;
    int n_idx = 0;
    for(int i=0; i<b.rows(); ++i)
    {
        if(std::abs(a-b(i))<0.01*b.cwiseAbs().maxCoeff())  //std::max(std::abs(b)))
        {
            out.push_back(i);
            n_idx++;
        }
    }
    Eigen::ArrayXi out_vec(n_idx);
    for(int i=0; i<n_idx; ++i)
    {
        out_vec(i) = out[i];
    }
    return(out_vec);
}

void build_V_list_from_master(std::vector<Eigen::MatrixXd> & V, const Eigen::MatrixXd & masterV, const std::vector<std::vector<int>> & map_idxs, int n, int k, int t)
{
    for(int i=0; i<n; ++i)
    {
        int kt = map_idxs[i].size();
        Eigen::MatrixXd Vi(kt,kt);
        V_assemble_IP(masterV,Vi,map_idxs[i],kt);
        V[i] = Vi;
    }
}

void V_assemble_IP(const Eigen::Ref<const Eigen::MatrixXd> & masterV, 
                   Eigen::MatrixXd & V_out,
                   const std::vector<int> & map_idxs, int kt)
{
    V_out.setZero(kt,kt);
    int sz = map_idxs.size();
    for(int r = 0; r < sz; ++r) {
        for(int c = 0; c < sz; ++c) {
            V_out(r, c) = masterV(map_idxs[r], map_idxs[c]);
        }
    }
}


Eigen::MatrixXd covCalc(const Eigen::MatrixXd & X)
{
    Eigen::MatrixXd centeredX = X.rowwise() - X.colwise().mean();
    Eigen::MatrixXd cov = (centeredX.adjoint() * centeredX) / double(X.rows());
    return(cov);
}

// [[Rcpp::export]]
Eigen::MatrixXd covCalc(const Eigen::MatrixXd & X, const Eigen::MatrixXi & MAP)
{
    int n = X.rows();
    int p = X.cols();
    int full_kt = MAP.cols();

    Eigen::MatrixXd SumXY = Eigen::MatrixXd::Zero(p, p);
    Eigen::MatrixXd N = Eigen::MatrixXd::Zero(p, p);
    Eigen::MatrixXd SumX_shared = Eigen::MatrixXd::Zero(p, p);

    for(int i = 0; i < n; ++i)
    {
        // 1. Collect non-zero indices to skip the vast majority of the empty matrix
        std::vector<int> active;
        active.reserve(p);
        // DYNAMIC MASKING: Prevent Out-of-Bounds Memory Reads
        if (p == full_kt) {
            // Phase 1: Residual Covariance (X is n x kt)
            for(int c = 0; c < p; ++c) {
                if(MAP(i, c) != 0) active.push_back(c);
            }
        } else {
            // Phase 2: Random Effects Covariance (X is n x 2k)
            int timepts = full_kt / (p / 2);
            for(int c = 0; c < p; ++c) {
                int node = c / 2; // Map slope/intercept back to the node
                bool has_data = false;
                for(int time = 0; time < timepts; ++time) {
                    if(MAP(i, node * timepts + time) != 0) {
                        has_data = true; break;
                    }
                }
                if(has_data) active.push_back(c);
            }
        }

        // 2. Only loop over active observation pairs!
        for(size_t idx1 = 0; idx1 < active.size(); ++idx1)
        {
            int c1 = active[idx1];
            double x1 = X(i, c1);
            for(size_t idx2 = 0; idx2 <= idx1; ++idx2)
            {
                int c2 = active[idx2];
                double x2 = X(i, c2);

                SumXY(c1, c2) += x1 * x2;
                N(c1, c2) += 1.0;
                SumX_shared(c1, c2) += x1;
                if(c1 != c2) {
                    SumX_shared(c2, c1) += x2;
                }
            }
        }
    }

    Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(p,p);
    for(int i=0; i < p; ++i)
    {
        for(int j=0; j <= i; ++j)
        {
            double Nij = N(i,j);
            if(Nij > 1.0)
            {
                double val = (SumXY(i, j) - (SumX_shared(i, j) * SumX_shared(j, i) / Nij)) / Nij;
                cov(i,j) = val;
                if(i != j) cov(j,i) = val;
            }
        }
    }
    return cov;
}

Eigen::MatrixXd covCalc(const Eigen::MatrixXd & X, const std::vector<std::vector<int>> & map_idxs, int p)
{
    int n = X.rows();
    Eigen::MatrixXd SumXY = Eigen::MatrixXd::Zero(p, p);
    Eigen::MatrixXd N = Eigen::MatrixXd::Zero(p, p);
    Eigen::MatrixXd SumX_shared = Eigen::MatrixXd::Zero(p, p);

    for(int i = 0; i < n; ++i)
    {
        // Instantly grab the precomputed active list!
        const std::vector<int>& active = map_idxs[i];
        
        for(size_t idx1 = 0; idx1 < active.size(); ++idx1)
        {
            int c1 = active[idx1];
            double x1 = X(i, c1);
            for(size_t idx2 = 0; idx2 <= idx1; ++idx2)
            {
                int c2 = active[idx2];
                double x2 = X(i, c2);

                SumXY(c1, c2) += x1 * x2;
                N(c1, c2) += 1.0;
                SumX_shared(c1, c2) += x1;
                if(c1 != c2) SumX_shared(c2, c1) += x2;
            }
        }
    }

    Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(p,p);
    for(int i=0; i < p; ++i) {
        for(int j=0; j <= i; ++j) {
            double Nij = N(i,j);
            if(Nij > 1.0) {
                double val = (SumXY(i, j) - (SumX_shared(i, j) * SumX_shared(j, i) / Nij)) / Nij;
                cov(i,j) = val;
                if(i != j) cov(j,i) = val;
            }
        }
    }
    return cov;
}


void covCalc(const Eigen::MatrixXd & X, const std::vector<std::vector<int>> & map_idxs, 
             Eigen::MatrixXd & cov, int p)
{
    int n = X.rows();
    Eigen::MatrixXd SumXY = Eigen::MatrixXd::Zero(p, p);
    Eigen::MatrixXd N = Eigen::MatrixXd::Zero(p, p);
    Eigen::MatrixXd SumX_shared = Eigen::MatrixXd::Zero(p, p);

    for(int i = 0; i < n; ++i)
    {
        // Instantly grab the precomputed active list!
        const std::vector<int>& active = map_idxs[i];
        
        for(size_t idx1 = 0; idx1 < active.size(); ++idx1)
        {
            int c1 = active[idx1];
            double x1 = X(i, c1);
            for(size_t idx2 = 0; idx2 <= idx1; ++idx2)
            {
                int c2 = active[idx2];
                double x2 = X(i, c2);

                SumXY(c1, c2) += x1 * x2;
                N(c1, c2) += 1.0;
                SumX_shared(c1, c2) += x1;
                if(c1 != c2) SumX_shared(c2, c1) += x2;
            }
        }
    }

    cov = Eigen::MatrixXd::Zero(p,p);
    for(int i=0; i < p; ++i) {
        for(int j=0; j <= i; ++j) {
            double Nij = N(i,j);
            if(Nij > 1.0) {
                double val = (SumXY(i, j) - (SumX_shared(i, j) * SumX_shared(j, i) / Nij)) / Nij;
                cov(i,j) = val;
                if(i != j) cov(j,i) = val;
            }
        }
    }
    return;
}

void get_cov_stats(const Eigen::Ref<const Eigen::MatrixXd>& R, 
                   const std::vector<std::vector<int>>& active_lists, 
                   const std::vector<int>& indices,
                   Eigen::MatrixXd& SumXY, Eigen::MatrixXd& N, 
                   Eigen::MatrixXd& SumX_shared, Eigen::MatrixXd& SumRsq)
{
    int p = R.cols();

    SumXY.setZero(p, p);
    N.setZero(p, p);
    SumX_shared.setZero(p, p);
    SumRsq.setZero(p, p);

    for(int row_idx : indices) 
    {
        // Instantly grab the precomputed active list
        const std::vector<int>& active = active_lists[row_idx];

        // 2. Accumulate raw sums for the active pairs
        for(size_t idx1 = 0; idx1 < active.size(); ++idx1) 
        {
            int c1 = active[idx1];
            double r1 = R(row_idx, c1);
            double rsq1 = r1 * r1;
            
            for(size_t idx2 = 0; idx2 <= idx1; ++idx2) 
            {
                int c2 = active[idx2];
                double r2 = R(row_idx, c2);
                double rsq2 = r2 * r2;

                SumXY(c1, c2) += r1 * r2;
                N(c1, c2) += 1.0;
                SumX_shared(c1, c2) += r1;
                SumRsq(c1, c2) += rsq1 * rsq2;
                
                if(c1 != c2) {
                    SumX_shared(c2, c1) += r2;
                }
            }
        }
    }
}

void build_cov_and_theta(const Eigen::MatrixXd& SumXY, const Eigen::MatrixXd& N, 
                         const Eigen::MatrixXd& SumX_shared, const Eigen::MatrixXd& SumRsq,
                         Eigen::MatrixXd& cov, Eigen::ArrayXXd& theta)
{
    int p = SumXY.cols();
    cov = Eigen::MatrixXd::Zero(p, p);
    theta = Eigen::ArrayXXd::Zero(p, p);

    for(int i = 0; i < p; ++i) 
    {
        for(int j = 0; j <= i; ++j) 
        {
            double Nij = N(i, j);
            if(Nij > 1.0) 
            {
                // 1. Covariance Formula
                double val = (SumXY(i, j) - (SumX_shared(i, j) * SumX_shared(j, i) / Nij)) / Nij;
                cov(i, j) = val;
                
                // 2. R^T R Formula
                double rtr_val = SumRsq(i, j) / Nij;
                
                // 3. Theta (Variance) Formula -> max(0.0) protects against microscopic float noise
                double theta_val = std::max(0.0, rtr_val - (val * val));
                theta(i, j) = std::sqrt(theta_val);
                
                // Mirror to upper triangle
                if(i != j) {
                    cov(j, i) = val;
                    theta(j, i) = theta(i, j);
                }
            }
        }
    }
}

void vec2list(const std::vector<Eigen::MatrixXd>& vec, Rcpp::List & out)
{
    for(int i=0;i<vec.size();++i) //(Eigen::MatrixXd i:vec)
    {
        out[i] = vec[i];//out.push_back(i);
    }
}

void list2vec(std::vector<Eigen::MatrixXd>& vec, const Rcpp::List & list)
{
    if(list.size() != vec.size())
    {
        Rcpp::Rcout << "VEC SIZE (" << vec.size() << ") INCOMPATIBLE W/ LIST SIZE (" << list.size() << ")\n"; 
    }
    for(int i=0;i<list.size();++i) //(Eigen::MatrixXd i:vec)
    {
        vec[i] = list[i];//out.push_back(i);
    }
}

void find_all(const std::vector<int> & vec, const int & val, std::vector<int> & out_val, std::vector<int> & out_not_val)
{
    out_val.reserve(vec.size());
    out_not_val.reserve(vec.size());
    for(int iter=0;iter<vec.size();++iter)
    {
        if(vec[iter] == val)
        {
            out_val.push_back(iter);
        } else
        {
            out_not_val.push_back(iter);
        }
    }
}

void find_all(const Eigen::VectorXi & vec, const int & val, std::vector<int> & out_val, std::vector<int> & out_not_val)
{
    out_val.reserve(vec.size());
    out_not_val.reserve(vec.size());
    for(int iter=0;iter<vec.size();++iter)
    {
        if(vec(iter) == val)
        {
            out_val.push_back(iter);
        } else
        {
            out_not_val.push_back(iter);
        }
    }
}


Eigen::MatrixXd Et_assemble(const Eigen::VectorXd & E, 
                            const Eigen::MatrixXi & MAP, 
                            int i, int k, int t, int kt)
{
    Eigen::MatrixXd Et = Eigen::MatrixXd::Zero(kt,kt);
    int cnt = 0;
    int cnt2 = 0;
    for(int j = 0; j < k; ++j)
    {
        for(int l = 0; l < t; ++l)
        {
        if(MAP(i,cnt2) == 1)
        {
            Et.diagonal()(cnt) = E(j);
            cnt++;
        }
        cnt2++;
        }
    }
  return(Et);
}

//[[Rcpp::export]]
Eigen::MatrixXd Z_assemble(const Eigen::MatrixXd & masterZ, 
                           const Eigen::MatrixXi & MAP,
                           int i, int k, int t, int kt)
{
    Eigen::MatrixXd Z_out = Eigen::MatrixXd::Zero(kt,2*k);
    int cnt = 0;
    for(int j = 0; j<k*t; ++j)
    {
        if(MAP(i,j) == 1)
        {
            Z_out.row(cnt).noalias() = masterZ.row(j);
            cnt++;
        }
    }
    return(Z_out);
}

void Et_assemble_IP(const Eigen::Ref<const Eigen::VectorXd> & E, 
                    Eigen::VectorXd & Et,
                    const std::vector<int> & map_idxs, 
                    int k, int t)
{
    int kt = map_idxs.size();
    Et.resize(kt);
    int cnt = 0;
    int cnt2 = 0;
    for(int j = 0; j < k; ++j)
    {
        for(int l = 0; l < t; ++l)
        {
            if(std::find(map_idxs.begin(), map_idxs.end(), cnt2) != map_idxs.end())
            {
                Et(cnt) = E(j);
                cnt++;
            }
            cnt2++;
        }
    }
}

void Z_assemble_IP(const Eigen::Ref<const Eigen::MatrixXd> & masterZt, 
                   Eigen::MatrixXd & Zt_out,
                   const std::vector<int> & map_idxs,
                   int k)
{
    // Function body remains identical
    int sz = map_idxs.size();
    Zt_out.resize(2*k,sz);
    //int cnt = 0;
    
    for(int j = 0; j<sz; ++j)
    {
        Zt_out.col(j).noalias() = masterZt.col(map_idxs[j]);
    }
}

void estimate_beta(const Eigen::MatrixXd & X, const Eigen::VectorXd & y, 
                   const Eigen::VectorXi kt_vec, const std::vector<std::vector<int>> & map_idxs,
                   const Eigen::MatrixXd & masterV, Eigen::VectorXd & beta,
                   int n, int k, int t, bool verbose, double eigen_threshold)
{
    int q = X.cols();
    Eigen::MatrixXd XVX = Eigen::MatrixXd::Zero(q,q);
    Eigen::VectorXd XVy = Eigen::VectorXd::Zero(q);
    Eigen::MatrixXd Vi(k*t,k*t);//, Xi;
    //Eigen::VectorXd yi;

    int cnt = 0;
    for(int i=0; i<n; ++i)
    {
        int kt = kt_vec(i);
        if(kt == 0) continue;
        V_assemble_IP(masterV,Vi,map_idxs[i],kt);

        // Map the current subject's X and y
        auto Xi = X.block(cnt, 0, kt, q);
        auto yi = y.segment(cnt, kt);

        Eigen::LLT<Eigen::MatrixXd> llt_Vi(Vi);
        if(llt_Vi.info() == Eigen::Success)
        {
            XVX.noalias() += Xi.transpose() * llt_Vi.solve(Xi);
            XVy.noalias() += Xi.transpose() * llt_Vi.solve(yi);
        }
        else{ // fallback if Vi not invertible
            Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod_Vi;
            cod_Vi.setThreshold(eigen_threshold);
            
            cod_Vi.compute(Vi);
            //Eigen::MatrixXd Vi_inv = cod_Vi.pseudoInverse();
            // Reconstruct the safely inverted matrix
            //Eigen::MatrixXd Vi_inv = evecs * evals.asDiagonal() * evecs.transpose();

            XVX.noalias() += Xi.transpose() * cod_Vi.solve(Xi);//Vi_inv * Xi;
            XVy.noalias() += Xi.transpose() * cod_Vi.solve(yi);//Vi_inv * yi;
        }
        cnt += kt;
    }
    Eigen::LDLT<Eigen::MatrixXd> ldlt_XVX(XVX);
    if(ldlt_XVX.info() == Eigen::Success)
    {
        beta = ldlt_XVX.solve(XVy);
    }
    else
    {
        if(verbose) {
            Rcpp::Rcout << "Warning: XVX matrix is not positive definite. Using pseudo-inverse fallback for beta estimation.\n";
        }
        beta = XVX.completeOrthogonalDecomposition().pseudoInverse() * XVy;
    }
    if(verbose)
    {
        Rcpp::Rcout << "XVX" << printdims(XVX) << std::endl;
        Rcpp::Rcout << XVX.block(0,0,5,5) << std::endl;
    }

}

void estimate_beta2(const Eigen::Ref<const Eigen::MatrixXd> & X, 
                    const Eigen::Ref<const Eigen::VectorXd> & y, 
                    const Eigen::Ref<const Eigen::MatrixXd> & Zt,
                    const Eigen::Ref<const Eigen::MatrixXd> & D,
                    const Eigen::Ref<const Eigen::VectorXd> & E,
                    const std::vector<std::vector<int>> & map_idxs_kt,
                    Eigen::VectorXd & beta,
                    int n, int k, int t)
{
    int q = X.cols();
    Eigen::MatrixXd XVX = Eigen::MatrixXd::Zero(q,q);
    Eigen::VectorXd XVy = Eigen::VectorXd::Zero(q);
    /*Eigen::MatrixXd D_inv;
    Eigen::LDLT<Eigen::MatrixXd> ldlt_D(D);
    if(ldlt_D.info() == Eigen::Success) {
        D_inv = ldlt_D.solve(Eigen::MatrixXd::Identity(2*k, 2*k));
    } else {
        D_inv = D.completeOrthogonalDecomposition().pseudoInverse();
    }*/
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(D);
    Eigen::VectorXd evals = es.eigenvalues().cwiseMax(0.0); // Mathematically enforce non-negative
    Eigen::MatrixXd L = es.eigenvectors() * evals.cwiseSqrt().asDiagonal(); // D = LL^T

    Eigen::MatrixXd Zti, ZiX, M;
    Eigen::VectorXd E_inv, Ziy;
    int cnt = 0;
    
    for(int i = 0; i < n; ++i)
    {
        int kt = map_idxs_kt[i].size();//kt_vec(i);
        
        // Get Z for this subject
        Z_assemble_IP(Zt, Zti, map_idxs_kt[i],k);
        
        // Bypass creating a dense Et matrix. Just grab the inverted diagonal
        E_inv.resize(kt);
        int c = 0;
        for(int idx : map_idxs_kt[i]) {
            E_inv(c++) = 1.0 / E(idx / t); 
        }
        
        // Map the current subject's X and y
        auto Xi = X.block(cnt, 0, kt, q);
        auto yi = y.segment(cnt, kt);
        
        // The Woodbury Transformation Variables
        Eigen::MatrixXd X_tilde = E_inv.asDiagonal() * Xi;
        Eigen::VectorXd y_tilde = E_inv.asDiagonal() * yi;
        Eigen::MatrixXd Ui = Zti.transpose() * L;
        Eigen::MatrixXd Ui_tilde = E_inv.asDiagonal() * Ui;
        
        // Accumulate the base E^-1 terms
        XVX.noalias() += Xi.transpose() * X_tilde;
        XVy.noalias() += Xi.transpose() * y_tilde;
    
        
        // Inner matrix M = D^-1 + Z^T * E^-1 * Z  (Always exactly 2k x 2k)
        M = Eigen::MatrixXd::Identity(2*k, 2*k) + Ui.transpose() * Ui_tilde; 

        // Inner Woodbury components
        Eigen::MatrixXd UiX = Ui_tilde.transpose() * Xi;  
        Eigen::VectorXd Uiy = Ui_tilde.transpose() * yi;  
        
        // Fast 2k x 2k decomposition
        Eigen::LLT<Eigen::MatrixXd> llt_M(M);
        if(llt_M.info() == Eigen::Success)
        {
            XVX.noalias() -= UiX.transpose() * llt_M.solve(UiX);
            XVy.noalias() -= UiX.transpose() * llt_M.solve(Uiy);
        }
        else
        {
            Eigen::MatrixXd M_inv = M.completeOrthogonalDecomposition().pseudoInverse();
            XVX.noalias() -= UiX.transpose() * M_inv * UiX;
            XVy.noalias() -= UiX.transpose() * M_inv * Uiy;
        }
        
        cnt += kt;
    }
    // Ensure XVX is symmetric
    XVX = 0.5 * (XVX + XVX.transpose()).eval();
    // Final beta solve outside the loop
    Eigen::LDLT<Eigen::MatrixXd> ldlt_XVX(XVX);
    if(ldlt_XVX.info() == Eigen::Success)
    {
        beta = ldlt_XVX.solve(XVy);
    }
    else
    {
        beta = XVX.completeOrthogonalDecomposition().pseudoInverse() * XVy;
    }
}

Eigen::VectorXd R_expand(const Eigen::VectorXd & R,
                         const Eigen::MatrixXi & MAP,
                         int idx, int q)
{
    Eigen::VectorXd R_out = Eigen::VectorXd::Zero(q);
    int cnt = 0;
    for(int i=0;i<q;++i)
    {
        if(MAP(idx,i) == 1)
        {
            R_out(i) = R(cnt);
            cnt++;
        }
    }
    return(R_out);
}

Eigen::MatrixXd RtR(const Eigen::MatrixXd & R, const std::vector<std::vector<int>>& active_lists)
{
    int n = R.rows();
    int p = R.cols();

    Eigen::MatrixXd SumRsq = Eigen::MatrixXd::Zero(p, p);
    Eigen::MatrixXd N = Eigen::MatrixXd::Zero(p, p);

    for(int i = 0; i < n; ++i)
    {
        // Instantly grab the precomputed active list
        const std::vector<int>& active = active_lists[i];

        for(size_t idx1 = 0; idx1 < active.size(); ++idx1)
        {
            int c1 = active[idx1];
            double rsq1 = R(i, c1) * R(i, c1); 
            
            for(size_t idx2 = 0; idx2 <= idx1; ++idx2)
            {
                int c2 = active[idx2];
                double rsq2 = R(i, c2) * R(i, c2);

                SumRsq(c1, c2) += rsq1 * rsq2;
                N(c1, c2) += 1.0;
            }
        }
    }

    Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(p, p);
    for(int i=0; i < p; ++i)
    {
        for(int j=0; j <= i; ++j)
        {
            if(N(i,j) > 0.0) 
            {
                double val = SumRsq(i, j) / N(i,j);
                cov(i, j) = val;
                if(i != j) cov(j, i) = val;
            }
        }
    }
    return cov;
}

// Helper to calculate the upper/lower bounds safely
void get_bounds(const Eigen::MatrixXd& cov, const Eigen::ArrayXXd& theta, double& lower, double& upper) 
{
    Eigen::ArrayXXd safe_theta = (theta == 0.0).select(1e-8, theta);
    Eigen::MatrixXd delta = (cov.array() / safe_theta).cwiseAbs().matrix();
    delta.diagonal() = Eigen::VectorXd::Zero(delta.rows());
    
    upper = delta.maxCoeff();
    lower = (delta.array() <= 0.0).select(std::numeric_limits<double>::max(), delta).minCoeff();
    if (lower == std::numeric_limits<double>::max()) lower = 0.0;
}

// [[Rcpp::export]]
bool check_openmp() {
#ifdef _OPENMP
    return true;
#else
    return false;
#endif
}