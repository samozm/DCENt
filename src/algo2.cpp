#define STRICT_R_HEADERS
#include "utils.h"
#include <chrono>

/*  
******************************************************************************
******************************************************************************
************ EVERY SUBJECT MUST HAVE ALL NODES AT AT LEAST 1 TIME ************
************                 OR AT LEAST A COLUMN                 ************
************         (ALL MATRICES NEED TO HAVE WIDTH 2K)         ************
******************************************************************************
******************************************************************************
*/

Eigen::VectorXd Zbcalc(const Eigen::MatrixXd & Zt, 
                       const Eigen::VectorXd & b, 
                       const std::vector<std::vector<int>> & map_idxs_kt, 
                       int n, int k, int t, int nkt)
{
    Eigen::VectorXd Zb = Eigen::VectorXd::Zero(nkt);
    Eigen::MatrixXd Zti(k*t,2*k);
    int cnt = 0;
    for(int i = 0; i < n; ++i)
    {
        int kt = map_idxs_kt[i].size();
        if(kt == 0) continue;
        Z_assemble_IP(Zt,Zti,map_idxs_kt[i],k);
        Zb.segment(cnt,kt) = Zti.transpose() * b; 
        cnt += kt;
    }
    return(Zb);
}


void calc_b(const Eigen::MatrixXd & X, 
            const Eigen::VectorXd & r0, 
            const Eigen::MatrixXd & Zt,
            const Eigen::MatrixXd & Lambda_D, 
            const Eigen::VectorXd & E, 
            Eigen::VectorXd & b,
            const std::vector<std::vector<int>> & map_idxs_kt,
            int n, int k, int t, int nkt)
{
    int q = 2*k;
    Eigen::MatrixXd ZEEZ = Eigen::MatrixXd::Zero(q,q);
    //Eigen::MatrixXd DZETE = Eigen::MatrixXd::Zero(q,nkt);
    Eigen::VectorXd DZETEr0 = Eigen::VectorXd::Zero(q);
    Eigen::VectorXd EInv = E.array().inverse(); //E.inverse();
    Eigen::MatrixXd Zti(q,k*t), EZt(q,k*t);
    Eigen::VectorXd EtInvr0(k*t), ZiEtInvr0(q),EtInv(k*t);
    int cnt = 0;
    for(int i=0;i<n;++i)
    {
        int kt = map_idxs_kt[i].size();
        if(kt == 0) continue;
        EZt.resize(q,kt);
        EtInvr0.resize(kt);
        //Et_Z_assemble_IP(EInv,EtInv,Zt,Zti,MAP,i,k,t,kt);
        Et_assemble_IP(EInv,EtInv,map_idxs_kt[i],k,t);
        Z_assemble_IP(Zt,Zti,map_idxs_kt[i],k);

        Eigen::RowVectorXd EInvRow = EtInv.transpose();
        EZt = Zti* EInvRow.asDiagonal();
        ZEEZ.noalias() += EZt * EZt.transpose();
        
        // 4. Vectorized square and multiply (No dense matrix math)
        EtInvr0 = EInvRow.array().square().transpose() * r0.segment(cnt, kt).array();
        
        ZiEtInvr0.noalias() = Zti * EtInvr0;
        DZETEr0.noalias() += Lambda_D.transpose() * ZiEtInvr0;
        cnt += kt;
    }

    Eigen::MatrixXd DtZEEZD = Eigen::MatrixXd::Identity(q,q);
    DtZEEZD.noalias() += Lambda_D.transpose() * ZEEZ * Lambda_D;
    Eigen::MatrixXd DtZEEZDDZETEr0 = DtZEEZD.llt().solve(DZETEr0);
    b = Lambda_D.transpose() * DtZEEZDDZETEr0;
}

void estimate_E(const Eigen::MatrixXd & X, 
                const Eigen::VectorXd & r0, 
                const Eigen::MatrixXd & Zt,
                const Eigen::MatrixXd & Lambda_D, 
                Eigen::VectorXd & Lambda_E,
                const std::vector<std::vector<int>> & map_idxs_kt, 
                int n, int k, int t, int nkt)
{
    Eigen::VectorXd b;
    calc_b(X,r0,Zt,Lambda_D,Lambda_E.array().square(),b,map_idxs_kt,n,k,t,nkt);
    Eigen::VectorXd r = r0 - Zbcalc(Zt, b, map_idxs_kt, n, k, t, nkt);
    //Eigen::MatrixXd R = Eigen::MatrixXd::Zero(n*t,k);
    Eigen::VectorXd sum_val = Eigen::VectorXd::Zero(k);
    Eigen::VectorXd sum_sq = Eigen::VectorXd::Zero(k);
    Eigen::VectorXi counts = Eigen::VectorXi::Zero(k);
    int current = 0;
    for(int i=0; i<n; ++i)
    {
        std::vector<int> map_idxs = map_idxs_kt[i];
        for(int j=0; j<k; ++j)
        {
            for(int l=0; l<t; ++l)
            {
                if(std::find(map_idxs.begin(), map_idxs.end(), j*t+l) != map_idxs.end())
                {
                    //R(i*t + l,j) = r(current);
                    double val = r(current++);
                    sum_val(j) += val;
                    sum_sq(j) += val*val;
                    counts(j)++;
                }
            }
        }
    }
    Eigen::ArrayXd means = sum_val.array() / counts.array().cast<double>();
    Eigen::ArrayXd mean_sq = sum_sq.array() / counts.array().cast<double>();
    
    // Update Lambda_E (standard deviation)
    Lambda_E = (mean_sq - means.square()).max(1e-4).sqrt();
}

void calc_e(const Eigen::VectorXd & r0, 
            const Eigen::MatrixXd & Zt,
            const Eigen::VectorXd & E, 
            const Eigen::MatrixXd & Lambda_D, 
            const std::vector<std::vector<int>> & map_idxs_kt, 
            Eigen::VectorXd & e, 
            int n, int k, int t, int nkt, bool timings)
{
    int p = 2*k;
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(p,p);
    Eigen::MatrixXd ZEEZ = Eigen::MatrixXd::Zero(p,p);
    //Eigen::MatrixXd EZ = Eigen::MatrixXd::Zero(nkt,p);
    Eigen::MatrixXd Zti(p,k*t), EZt_tmp(p,k*t);
    Eigen::VectorXd E_tmp(k*t);
    Eigen::VectorXd Zr = Eigen::VectorXd::Zero(p);

    auto t1 = std::chrono::high_resolution_clock::now();
    int cnt = 0;
    for(int i=0;i<n;++i)
    {
        int kt = map_idxs_kt[i].size();
        if(kt == 0) continue;
        EZt_tmp.resize(p, kt);
        //Et_Z_assemble_IP(E,E_tmp,Zt,Zti,MAP,i,k,t,kt);
        Z_assemble_IP(Zt,Zti,map_idxs_kt[i],k);
        Et_assemble_IP(E,E_tmp,map_idxs_kt[i],k,t);
        B.noalias() += Zti * Zti.transpose();
        EZt_tmp.noalias() = Zti * E_tmp.asDiagonal();
        ZEEZ.noalias() += EZt_tmp * EZt_tmp.transpose();
        Zr.noalias() += Zti * r0.segment(cnt,kt);
        cnt += kt;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    Eigen::MatrixXd BD = B * Lambda_D;
    Eigen::MatrixXd BDBinv =  BD * BD.transpose();
    Eigen::MatrixXd BDBinvZEEZ =  BDBinv + ZEEZ;
    Eigen::LLT<Eigen::MatrixXd> BDBinvZEEZ_llt = BDBinvZEEZ.llt();
    //Eigen::MatrixXd HZEEZ(k*t,p);
    // Safe BDBinv solve
    Eigen::VectorXd BDBinvZr;
    Eigen::LDLT<Eigen::MatrixXd> ldlt_BDBinv(BDBinv);
    if(ldlt_BDBinv.info() == Eigen::Success) {
        BDBinvZr = ldlt_BDBinv.solve(Zr);
    } else {
        BDBinvZr = BDBinv.completeOrthogonalDecomposition().pseudoInverse() * Zr;
    }
    // Safe BDBinvZEEZ solve
    Eigen::MatrixXd ZEEZsolve;
    Eigen::LDLT<Eigen::MatrixXd> ldlt_BDBinvZEEZ(BDBinvZEEZ);
    if(ldlt_BDBinvZEEZ.info() == Eigen::Success) {
        ZEEZsolve = ldlt_BDBinvZEEZ.solve(ZEEZ);
    } else {
        ZEEZsolve = BDBinvZEEZ.completeOrthogonalDecomposition().pseudoInverse() * ZEEZ;
    }

    Eigen::VectorXd BDBinvZrZEEZsolveBDBinvZr = BDBinvZr - (ZEEZsolve * BDBinvZr);
    Eigen::VectorXd ZtiBDBinvZrZEEZsolveBDBinvZr(k*t);
    auto t3 = std::chrono::high_resolution_clock::now();
    cnt = 0; 
    for(int i=0;i<n;++i)
    {
        int kt = map_idxs_kt[i].size();
        if(kt == 0) continue;
        ZtiBDBinvZrZEEZsolveBDBinvZr.resize(kt);
        //Et_Z_assemble_IP(E,E_tmp,Zt,Zti,MAP,i,k,t,kt);
        Et_assemble_IP(E,E_tmp,map_idxs_kt[i],k,t);
        Z_assemble_IP(Zt,Zti,map_idxs_kt[i],k);
        ZtiBDBinvZrZEEZsolveBDBinvZr = Zti.transpose() * BDBinvZrZEEZsolveBDBinvZr;
        e.segment(cnt,kt).array() = ZtiBDBinvZrZEEZsolveBDBinvZr.array() * E_tmp.array().square();
        cnt += kt;
    }
    auto t4 = std::chrono::high_resolution_clock::now();
    if(timings)
    {
        double time_ZEEZ = std::chrono::duration<double, std::milli>(t2 - t1).count();
        double time_threshold    = std::chrono::duration<double, std::milli>(t3 - t2).count();
        double time_bailout    = std::chrono::duration<double, std::milli>(t4 - t3).count();

        Rcpp::Rcout << "build ZEEZ:                  " << time_ZEEZ << " ms\n";
        Rcpp::Rcout << "BDBinvZrZEEZsolveBDBinvZr:   " << time_threshold << " ms\n";
        Rcpp::Rcout << "build e:                     " << time_bailout << " ms\n";
    }
}

void a2_thresholdRange(const Eigen::MatrixXd & R, Eigen::ArrayXXd& theta, Eigen::MatrixXd& cov, 
                    const std::vector<std::vector<int>> & map_idxs_2k, double & lower, double & upper)
{
    int n = R.rows();
    int p = R.cols();
    covCalc(R,map_idxs_2k,cov,p);

    // Prevent NaN from sqrt(negative floating point noise)
    theta = (RtR(R, map_idxs_2k) - cov.array().square().matrix()).array().max(0.0).sqrt();
    
    Eigen::ArrayXXd safe_theta = (theta == 0.0).select(1e-8, theta);
    Eigen::MatrixXd delta = (cov.array() / safe_theta).cwiseAbs().matrix();
    delta.diagonal() = Eigen::VectorXd::Zero(delta.rows());
    
    upper = delta.maxCoeff();
    lower = (delta.array() <= 0.0).select(std::numeric_limits<double>::max(), delta).minCoeff();
    if (lower == std::numeric_limits<double>::max()) lower = 0.0;
}

void a2_threshold(const Eigen::MatrixXd& abscov, const Eigen::MatrixXd& signcov, double lambda,
               const Eigen::ArrayXXd& theta, Eigen::MatrixXd& sigma_out)
{
    int p = abscov.rows();
    
    // Loop through the matrix without allocating any temporary arrays
    for(int i = 0; i < p; ++i) 
    {
        double diag_diff_i = abscov(i,i) - (theta(i,i) * lambda);
        for(int j = 0; j < p; ++j) 
        {
            double diag_diff_j = abscov(j,j) - (theta(j,j) * lambda);
            // Apply the diagonal mask and threshold in one step
            if (diag_diff_i > 0.0 && diag_diff_j > 0.0) 
            {
                double val = abscov(i, j) - theta(i, j) * lambda;
                sigma_out(i, j) = std::max(0.0, val) * signcov(i, j);
            } 
            else 
            {
                sigma_out(i, j) = 0.0;
            }
        }
        sigma_out(i, i) = std::max(0.0, diag_diff_i);
    }
}


void a2_threshold_D(const Eigen::MatrixXd & R, Eigen::MatrixXd& sigma, 
                 Eigen::ArrayXXd & theta, const std::vector<std::vector<int>> & map_idxs_2k, 
                 int n_fold=5,int seed=1234, bool timings=false)
{
    auto rng = std::default_random_engine(seed);

    int n = R.rows();
    int p = R.cols();

    // Dynamic Fold Adjustment for Subsets
    int actual_folds = std::min(n_fold, n);
    
    // Bailing out safely if the subset is extremely small (1 subject)
    if (actual_folds < 2) {
        Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(p,p);
        covCalc(R, map_idxs_2k, cov, p);
        Eigen::MatrixXd covAbs = cov.cwiseAbs();
        Eigen::MatrixXd covSign = cov.cwiseSign();
        a2_threshold(covAbs, covSign, 1.0, theta, sigma);
        return; 
    }

    int nParam = 100;
    Eigen::MatrixXd cov(p,p);
    double lower = 0.0;
    double upper = 0.0;
    sigma.setZero(p,p);
    
    // 1. Pre-calculate the TOTAL dataset aggregates ONCE
    std::vector<int> all_idx(n);
    std::iota(all_idx.begin(), all_idx.end(), 0);
    
    Eigen::MatrixXd SumXY_tot, N_tot, SumX_shared_tot, SumRsq_tot;
    get_cov_stats(R, map_idxs_2k, all_idx, SumXY_tot, N_tot, SumX_shared_tot, SumRsq_tot);
    // Build the master baseline cov and theta
    build_cov_and_theta(SumXY_tot, N_tot, SumX_shared_tot, SumRsq_tot, cov, theta);
    get_bounds(cov, theta, lower, upper);

    std::vector<double> params(nParam);
    double jump = (upper - lower)/double(nParam);
    double ctr = lower;
    std::generate(params.begin(), params.end(), [&ctr,&jump]{ return ctr+=jump;});
    
    std::vector<int> part(n);
    std::iota(part.begin(), part.end(), 0);
    std::shuffle(part.begin(),part.end(), rng);

    Eigen::MatrixXd error = Eigen::MatrixXd::Zero(actual_folds,nParam);
    Eigen::MatrixXd covTest;
    Eigen::MatrixXd covTrain(p,p);
    Eigen::ArrayXXd thetaTrain(p,p);

    for (int i=0;i<part.size();++i)
    {
        part[i] = part[i] % actual_folds;
    }
    for (int i=0;i<actual_folds; ++i)
    {
        std::vector<int> val_idx;
        std::vector<int> not_val_idx; // We keep this for find_all, but we NEVER use it!
        find_all(part, i, val_idx, not_val_idx);
        
        // A. Process ONLY the tiny 20% test fold
        Eigen::MatrixXd SumXY_test, N_test, SumX_shared_test, SumRsq_test;
        get_cov_stats(R, map_idxs_2k, val_idx, SumXY_test, N_test, SumX_shared_test, SumRsq_test);
        
        // B. THE TRICK: Derive the 80% train fold instantly via subtraction!
        Eigen::MatrixXd SumXY_train = SumXY_tot - SumXY_test;
        Eigen::MatrixXd N_train = N_tot - N_test;
        Eigen::MatrixXd SumX_shared_train = SumX_shared_tot - SumX_shared_test;
        Eigen::MatrixXd SumRsq_train = SumRsq_tot - SumRsq_test;
        
        // C. Build the matrices
        Eigen::ArrayXXd dummy_thetaTest; // Not used, just needed for the function
        build_cov_and_theta(SumXY_test, N_test, SumX_shared_test, SumRsq_test, covTest, dummy_thetaTest);
        build_cov_and_theta(SumXY_train, N_train, SumX_shared_train, SumRsq_train, covTrain, thetaTrain);
        
        // D. OpenMP Loop
        std::vector<Eigen::MatrixXd> local_sigmas(nParam, Eigen::MatrixXd::Zero(p, p));
        Eigen::MatrixXd covTrainAbs = covTrain.cwiseAbs();
        Eigen::MatrixXd covTrainSign = covTrain.cwiseSign();

        #pragma omp parallel for
        for(int j=0;j<nParam;++j)
        {
            a2_threshold(covTrainAbs,covTrainSign,params[j],thetaTrain,local_sigmas[j]);
            // Calculate the norm manually. (local_sigmas - covTest).norm() 
            // forces Eigen to create a temporary matrix, which would crash R!
            double sq_err = 0.0;
            for(int r = 0; r < p; ++r) {
                for(int c = 0; c < p; ++c) {
                    double diff = local_sigmas[j](r,c) - covTest(r,c);
                    sq_err += diff * diff;
                }
            }
            // Catch NaN or Infinity before it infects the error matrix
            if (std::isnan(sq_err) || std::isinf(sq_err)) 
            {
                // Penalize this threshold heavily so it is never picked as the minimum,
                // but keep it as a finite double so it doesn't break the column sum!
                error(i,j) = std::numeric_limits<double>::max() / 10000.0; 
            } 
            else 
            {
                error(i,j) = std::sqrt(sq_err);
            }
        }
    }
    Eigen::Index minIndex;
    error.colwise().sum().minCoeff(&minIndex);
    //theta = params[minIndex] * theta;

    Eigen::MatrixXd covAbs = cov.cwiseAbs();
    Eigen::MatrixXd covSign = cov.cwiseSign();
    a2_threshold(covAbs,covSign,params[minIndex],theta,sigma);
    auto t4 = std::chrono::high_resolution_clock::now();
    theta = params[minIndex] * theta;
}

void estimate_D(const Eigen::MatrixXd & X, 
                const Eigen::VectorXd & r0, 
                const Eigen::MatrixXd & Zt,
                const Eigen::VectorXd & E, 
                Eigen::MatrixXd & Lambda_D, 
                const std::vector<std::vector<int>> & map_idxs_kt, 
                const std::vector<std::vector<int>> & map_idxs_2k, 
                Eigen::MatrixXd & D,
                Eigen::ArrayXXd & theta,
                int n, int k, int t, int nkt, 
                int itr, int n_fold=5, 
                bool custom_theta=false, 
                bool timings=false,
                int seed=1111,
                bool soft=1, 
                double eigen_threshold=pow(10,-2))
{
    int p = 2*k;
    Eigen::VectorXd e = Eigen::VectorXd::Zero(nkt);
    auto t1 = std::chrono::high_resolution_clock::now();
    calc_e(r0,Zt,E,Lambda_D,map_idxs_kt,e,n,k,t,nkt,timings);
    auto t2 = std::chrono::high_resolution_clock::now();
    Eigen::MatrixXd R(n,p),Zti(p,k*t),ZiTZi(p,p); // = Eigen::MatrixXd::Zero(n,p);
    Eigen::VectorXd ZiTr(p);
    Eigen::VectorXd r = r0 - e;
    int cnt = 0;
    for(int i=0;i<n;++i)
    {
        int kt = map_idxs_kt[i].size();//kt_vec(i);//Z[i].rows();
        if(kt == 0) continue;
        Z_assemble_IP(Zt,Zti,map_idxs_kt[i],k);
        ZiTZi.noalias() = Zti * Zti.transpose();
        ZiTZi.diagonal().array() += 1e-8; // ridge in case a whole column is 0
        ZiTr.noalias() = Zti * r.segment(cnt,kt);
        // SAFE SOLVER:
        Eigen::LDLT<Eigen::MatrixXd> ldlt_ZiTZi(ZiTZi);
        if(ldlt_ZiTZi.info() == Eigen::Success) {
            R.row(i) = ldlt_ZiTZi.solve(ZiTr);
        } else {
            R.row(i) = (ZiTZi.completeOrthogonalDecomposition().pseudoInverse() * ZiTr);
        }
        cnt += kt;
    }

    //TODO: only change theta every 5 iterations???
    //give option to have user input threshold theta
    if(itr % 5 == 0 && !custom_theta)
    {
        a2_threshold_D(R,D,theta,map_idxs_2k,n_fold,seed,timings);
    }else{
        Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(p,p);
        covCalc(R, map_idxs_2k, cov, p);
        D.setZero(p,p);
        //TODO: sqrt log p/n??
        a2_threshold(cov.cwiseAbs(),cov.cwiseSign(),1.0,theta,D);
    }
    auto t3 = std::chrono::high_resolution_clock::now();

    Eigen::MatrixXd solver_input = D + Eigen::MatrixXd::Identity(p, p);

    // 2. Attempt Cholesky Decomposition (LLT) directly
    Eigen::LLT<Eigen::MatrixXd> llt(solver_input);
    
    // 3. If it fails (matrix is not Positive Definite), apply an iterative ridge
    if(llt.info() == Eigen::NumericalIssue)
    {
        //Project to Positive Definite Space (Eigenvalue Clipping)
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(solver_input);
        Eigen::VectorXd evals = es.eigenvalues();
        Eigen::MatrixXd evecs = es.eigenvectors();

        bool needs_projection = false;
        double min_variance = eigen_threshold; // The smallest variance we will allow

        for(int j = 0; j < evals.size(); ++j) 
        {
            // If the variance is negative or zero, clip it to the minimum
            if(evals(j) < min_variance) 
            {
                evals(j) = min_variance; 
                needs_projection = true;
            }
        }

        // Reconstruct the matrix only if we had to fix it
        if(needs_projection) 
        {
            solver_input = evecs * evals.asDiagonal() * evecs.transpose();
        }
        llt.compute(solver_input);
    }
    
    // 4. We are now guaranteed a valid Lower Triangular matrix
    Lambda_D = llt.matrixL();
    auto t4 = std::chrono::high_resolution_clock::now();
    if(timings)
    {
        double time_calc_e = std::chrono::duration<double, std::milli>(t2 - t1).count();
        double time_threshold    = std::chrono::duration<double, std::milli>(t3 - t2).count();
        double time_bailout    = std::chrono::duration<double, std::milli>(t4 - t3).count();

        Rcpp::Rcout << "calc_e:                 " << time_calc_e << " ms\n";
        Rcpp::Rcout << "threshold:              " << time_threshold << " ms\n";
        Rcpp::Rcout << "bailout & lambda_D:     " << time_bailout << " ms\n";
    }
}

double calc_sigma2(const Eigen::MatrixXd & Zt, 
                   const Eigen::MatrixXd & D, 
                   const Eigen::VectorXd & E, 
                   const std::vector<std::vector<int>> & map_idxs_kt,
                   const Eigen::VectorXd & r0, 
                   int n, int k, int t, int p, 
                   bool REML=false)
{
    double sigma2 = 0.0;
    Eigen::MatrixXd Lambda_V;
    int nkt = 0;
    Eigen::MatrixXd Zti(2*k,k*t), ZDZit(k*t,k*t), ZiD(k*t,2*k);
    Eigen::VectorXd Et(k*t);
    for(int i=0; i<n;++i)
    {
        int kt = map_idxs_kt[i].size();
        if (kt == 0) continue;
        //Et_Z_assemble_IP(E, Et, Zt,Zti,MAP, i, k, t, kt);
        Et_assemble_IP(E,Et,map_idxs_kt[i],k,t);
        Z_assemble_IP(Zt,Zti,map_idxs_kt[i],k);

        ZiD.resize(kt,2*k);
        ZDZit.resize(kt,kt);
        // broken up for stupid compiler reasons
        ZiD.noalias() = Zti.transpose() * D;
        ZDZit.noalias() = ZiD * Zti;
        ZDZit.diagonal().array() += Et.array();

        Eigen::LDLT<Eigen::MatrixXd> ldlt_V(ZDZit);
        auto r_sub = r0.segment(nkt, kt);
        
        if (ldlt_V.info() == Eigen::Success) {
            sigma2 += r_sub.dot(ldlt_V.solve(r_sub));
        } else {
            // Safe fallback if the matrix collapses
            sigma2 += r_sub.dot(ZDZit.completeOrthogonalDecomposition().pseudoInverse() * r_sub);
        }
        nkt += kt;
    }
    if(REML)
    {
        sigma2 = sigma2 / (nkt- p);
    } else {
        sigma2 = sigma2 / (nkt);
    }
    return(sigma2);
}

int a2_initial_estimates(const Eigen::MatrixXd & X, const Eigen::VectorXd & y, 
                       const Eigen::MatrixXd & Zt, 
                       const std::vector<std::vector<int>> & map_idxs_kt, 
                       Eigen::MatrixXd & Lambda_D, Eigen::MatrixXd & D, 
                       Eigen::VectorXd & Lambda_E, Eigen::VectorXd & beta, 
                       Eigen::VectorXd & r,
                       int n, int k, int t)
{
    int p = X.cols();
    int q = 2*k; 
    // Manual XtX and Xty to bypass Apple Accelerate crashes
    Eigen::MatrixXd XtX = Eigen::MatrixXd::Zero(p, p);
    Eigen::VectorXd Xty = Eigen::VectorXd::Zero(p);
    
    for(int i=0; i < X.rows(); ++i) {
        for(int c1=0; c1 < p; ++c1) {
            Xty(c1) += X(i, c1) * y(i);
            for(int c2=0; c2 <= c1; ++c2) {
                XtX(c1, c2) += X(i, c1) * X(i, c2);
            }
        }
    }
    
    // Mirror the lower triangle to upper
    for(int c1=0; c1 < p; ++c1) {
        for(int c2=c1+1; c2 < p; ++c2) {
            XtX(c1, c2) = XtX(c2, c1);
        }
    }
    // SAFE SOLVER
    Eigen::LDLT<Eigen::MatrixXd> ldlt_XtX(XtX);
    if(ldlt_XtX.info() == Eigen::Success) {
        beta = ldlt_XtX.solve(Xty);
    } else {
        beta = XtX.completeOrthogonalDecomposition().pseudoInverse() * Xty;
    }
    r = y - X * beta;
    Eigen::MatrixXd R(n,k*t);
    int nkt = 0;
    for(int i = 0; i<n; ++i)
    {
        int kt0 = 0; // Tracks how many valid nodes this specific subject has
        for(int idx : map_idxs_kt[i])
        {
            R(i, idx) = r(nkt + kt0);
            kt0++;
        }
        nkt += kt0;
    }
    Eigen::MatrixXd cov_int = Eigen::MatrixXd::Zero(k*t, k*t);
    covCalc(R, map_idxs_kt, cov_int, k*t);
    for(int i=0;i<k;++i)
    {
        Lambda_E(i) = (cov_int.diagonal().array()).segment(i*t,t).sqrt().mean() /2;
    }

    cov_int.diagonal() = cov_int.diagonal()/2;

    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(q,q); 
    Eigen::MatrixXd ZCZ = Eigen::MatrixXd::Zero(q,q);
    Eigen::MatrixXd Zti(2*k,k*t), cov_sub(k*t,k*t);
    for(int i=0; i<n;++i)
    {
        int kt = map_idxs_kt[i].size();
        if(kt == 0) continue;
        Z_assemble_IP(Zt,Zti,map_idxs_kt[i],k);
        B += Zti * Zti.transpose();
        cov_sub.resize(kt, kt);
        int r_idx = 0;
        for(int ri : map_idxs_kt[i]) {
            int c_idx = 0;
            for(int ci : map_idxs_kt[i]) {
                cov_sub(r_idx, c_idx) = cov_int(ri, ci);
                c_idx++;
            }
            r_idx++;
        }
        ZCZ += Zti * cov_sub * Zti.transpose();
    }
    // 1. Solve the first half: M = B^-1 * ZCZ
    Eigen::MatrixXd M = B.ldlt().solve(ZCZ);
    
    // 2. Solve the second half: B * D^T = M^T
    Eigen::MatrixXd D_transpose = B.ldlt().solve(M.transpose());
    
    // 3. Transpose back to get D
    D = D_transpose.transpose();
    Lambda_D = (D + Eigen::MatrixXd::Identity(q,q)).llt().matrixL();
    return nkt;
}


//' Estimate Covariance Matrices and Fixed Effects (C++ Backend)
//'
//' @description A high-performance, OpenMP-enabled C++ backend for iteratively estimating 
//' fixed effects (\code{beta}), random effect covariance (\code{D}), and residual variance (\code{E}) 
//' using block-coordinate descent and cross-validated thresholding.
//'
//' @param X Numeric matrix of fixed effect covariates.
//' @param y Numeric vector of the continuous response variable.
//' @param masterZ Master random effect matrix containing all timepoints used by any subject
//' @param MAP Integer matrix, n x kt describing which node/timepoint combinations each subject has
//' @param n Integer. Number of subjects.
//' @param k Integer. Number of nodes or spatial features.
//' @param t Integer. Number of time points per subject.
//' @param theta Numeric matrix. Initial thresholding parameters for cross-validation.
//' @param max_itr Integer. Maximum number of block-coordinate descent iterations. Default is 250.
//' @param convergence_cutoff Numeric. Change in error required to declare convergence. Default is 0.0001.
//' @param REML Logical. If TRUE, uses Restricted Maximum Likelihood for variance estimation. Default is FALSE.
//' @param verbose Logical. If TRUE, prints iteration-level errors and matrix updates to the console. Default is FALSE.
//' @param timings Logical. If TRUE, prints millisecond timings for internal C++ functions. Default is FALSE.
//' @param n_fold Integer. Number of folds for the internal cross-validation step. Default is 5.
//' @param custom_theta Logical. If TRUE, bypasses cross-validation and uses the user-provided \code{theta}. Default is FALSE.
//' @param n_threads Integer. Number of OpenMP threads to use. Set to 1 if parallelizing at the R level. Default is 1.
//' @param seed Integer. Random seed for ensuring reproducible cross-validation splits. Default is 1234.
//' 
//' @return A named list containing:
//' \itemize{
//'   \item \code{Sigma}: The final composite covariance matrix.
//'   \item \code{E}: The diagonal residual variance matrix.
//'   \item \code{D}: The thresholded random effects covariance matrix.
//'   \item \code{Lambda_E}: The standard deviation vector of the residuals.
//'   \item \code{beta}: The final fixed-effects coefficient vector.
//'   \item \code{n_iter}: The total number of iterations run.
//'   \item \code{all_err}: Vector of convergence errors across all iterations.
//'   \item \code{converged}: Logical indicating if the algorithm reached \code{convergence_cutoff} before \code{max_itr}.
//'   \item \code{sigma}: The estimated scaling variance parameter.
//'   \item \code{threshold}: The final threshold matrix selected by cross-validation.
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::List estimate_DEbeta(const Eigen::Map<Eigen::MatrixXd> X, 
                           const Eigen::Map<Eigen::VectorXd> y, 
                           const Eigen::Map<Eigen::MatrixXd> masterZ, 
                           const Eigen::Map<Eigen::MatrixXi> MAP,
                           int n, int k, int t,
                           Eigen::ArrayXXd theta,
                           int max_itr=250, 
                           double convergence_cutoff=0.0001,
                           bool REML=false,
                           bool verbose=false,
                           bool timings=false,
                           int n_fold=5,
                           bool custom_theta = false,
                           int n_threads = 1,
                           int seed=1234)
{
    // Protect C++ from R's NAs
    // TODO: better NA handling?
    if (X.hasNaN() || y.hasNaN() || masterZ.hasNaN()) {
        Rcpp::stop("Input matrix X or vector y contains NA/NaN values. Please remove them before running DCENt.");
    }

    // Lock Eigen to 1 thread permanently 
    Eigen::setNbThreads(1); 
    
    // Control OpenMP dynamically from R!
    #ifdef _OPENMP
        omp_set_num_threads(n_threads);
    #endif

    int p = X.cols();
    Eigen::MatrixXd Lambda_D(2*k,2*k);
    Eigen::MatrixXd D(2*k,2*k);
    Eigen::VectorXd Lambda_E(k);
    Eigen::VectorXd beta(p);
    Eigen::VectorXd r0;

    Eigen::MatrixXd masterZt = masterZ.transpose();

    std::vector<std::vector<int>> map_idxs_kt(n);
    std::vector<std::vector<int>> map_idxs_2k(n);
    
    for(int i = 0; i < n; ++i) {
        // 1. Standard Residual Covariance Mapping (size kt)
        for(int j = 0; j < k*t; ++j) {
            if(MAP(i, j) == 1) {
                map_idxs_kt[i].push_back(j);
            }
        }
        
        // 2. Random Effects Covariance Mapping (size 2k)
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
    
    int nkt = a2_initial_estimates(X,y,masterZt,map_idxs_kt,Lambda_D,D,Lambda_E,beta,r0,n,k,t);

    Eigen::VectorXi kt_vec = MAP.rowwise().sum();
    Eigen::VectorXd beta_prev;
    Eigen::MatrixXd D_prev;
    Eigen::VectorXd E_prev;
    
    double sigma2 = 0.0;
    double err = 10.0;
    double prev_err = 9.0;
    double double_prev_err = 8.0;
    int n_itr = 0;
    std::vector<double> all_err(max_itr);
    while (((err > convergence_cutoff) || (prev_err > convergence_cutoff)) && (n_itr < max_itr))
    {
        // Give control back to R to check for Esc/Ctrl+C
        Rcpp::checkUserInterrupt();

        auto start_loop = std::chrono::high_resolution_clock::now();
        double_prev_err = prev_err;
        prev_err = err;
        beta_prev = beta;
        auto t1 = std::chrono::high_resolution_clock::now();
        estimate_beta2(X,y,masterZt,D,Lambda_E.array().square(),map_idxs_kt,beta,n,k,t);
        auto t2 = std::chrono::high_resolution_clock::now();
        double err2 = (beta - beta_prev).squaredNorm() / beta_prev.squaredNorm(); 

        r0 = y - X * beta; 
        D_prev = Lambda_D;
        estimate_D(X,r0,masterZt,Lambda_E.array().square(),Lambda_D,map_idxs_kt,map_idxs_2k,D,theta,n,k,t,nkt,n_itr,n_fold,custom_theta,timings);
        auto t3 = std::chrono::high_resolution_clock::now();
        double err0 = (Lambda_D - D_prev).squaredNorm() / D_prev.squaredNorm();

        E_prev = Lambda_E;
        estimate_E(X,r0,masterZt,Lambda_D,Lambda_E,map_idxs_kt,n,k,t,nkt);
        auto t4 = std::chrono::high_resolution_clock::now();
        double err1 = (Lambda_E - E_prev).squaredNorm() / E_prev.squaredNorm();

        err = (err0 + err1 + err2)/3;
        all_err[n_itr] = err;
        n_itr++;

        if(timings)
        {
            double time_beta = std::chrono::duration<double, std::milli>(t2 - t1).count();
            double time_D    = std::chrono::duration<double, std::milli>(t3 - t2).count();
            double time_E    = std::chrono::duration<double, std::milli>(t4 - t3).count();
            
            Rcpp::Rcout << "--- Iteration " << n_itr << " Timings (ms) ---\n";
            Rcpp::Rcout << "estimate_beta2: " << time_beta << " ms\n";
            Rcpp::Rcout << "estimate_D:     " << time_D << " ms\n";
            Rcpp::Rcout << "estimate_E:     " << time_E << " ms\n";
        }

        if(verbose)
        {
            Rcpp::Rcout << "overall error after " << n_itr << " iterations: " << err << "\n";
            Rcpp::Rcout << "lambda D "  << printdims(Lambda_D) << err0 << " \n";
            Eigen::Index maxX, maxY;
            double max = (Lambda_D - D_prev).maxCoeff(&maxX,&maxY);
            Rcpp::Rcout <<  "Max: " << max << " (" << maxX << "," << maxY << ")\n";
            Rcpp::Rcout << Lambda_D(Eigen::seqN(maxX-3,6),Eigen::seqN(maxY-3,6)) << "\n\n";
            Rcpp::Rcout << D_prev(Eigen::seqN(maxX-3,6),Eigen::seqN(maxY-3,6)) << "\n"; 

            Rcpp::Rcout << "E " << err1 << " \n";
            Rcpp::Rcout << Lambda_E(Eigen::seqN(0,5)) << "\n\n";
            Rcpp::Rcout << E_prev(Eigen::seqN(0,5)) << "\n"; 

            Rcpp::Rcout << "beta " << err2 << " \n";
            Rcpp::Rcout << beta(Eigen::seqN(0,5)) << "\n\n";
            Rcpp::Rcout << beta_prev(Eigen::seqN(0,5)) << "\n"; 

            max = (beta - beta_prev).maxCoeff(&maxX,&maxY);
            Rcpp::Rcout <<  "Max: " << max << " (" << maxX << "," << maxY << ")\n";
            Rcpp::Rcout << beta(Eigen::seqN(maxX-3,6)) << "\n\n";

            Rcpp::Rcout << "sigma " << "\n" << sigma2 << "\n";
        }
    }
    Eigen::VectorXd E0 = Lambda_E.array().square();
    sigma2 = calc_sigma2(masterZt,D,E0,map_idxs_kt,r0,n,k,t,p,REML);
    D = D * sigma2;
    Eigen::VectorXd E = E0 * sigma2;

    bool converged = false;
    if(n_itr < max_itr) converged = true;

    Eigen::VectorXd Et(k*t);
    for(int j = 0; j < k; ++j) {
        for(int l = 0; l < t; ++l) {
            Et(j * t + l) = E(j);
        }
    }

    Eigen::MatrixXd Sigma = masterZ * D * masterZt;
    Sigma.diagonal().array() += Et.array();
    //Eigen::MatrixXd Sigma(nkt,nkt);
    //calc_ZDZ_plus_E(masterZt,D,E,Sigma,MAP,n,k,t,nkt);
    Eigen::VectorXd b;
    calc_b(X, r0, masterZt, Lambda_D, E, b, map_idxs_kt, n, k, t, nkt);

    return(Rcpp::List::create(Rcpp::Named("Sigma")=Sigma,
           Rcpp::Named("E") = E,Rcpp::Named("D") = D, 
           Rcpp::Named("Lambda_E") = Lambda_E, 
           Rcpp::Named("beta") = beta, 
           Rcpp::Named("n_iter") = n_itr,
           Rcpp::Named("all_err") = all_err,
           Rcpp::Named("converged") = converged, 
           Rcpp::Named("sigma") = sigma2,
           Rcpp::Named("threshold")=theta,
           Rcpp::Named("b") = b));
}