/*
 See R. Graglia, "Numerical Integration of the linear shape functions
 times the 3-d Green's function or its gradient on a plane triangle", IEEE
 Transactions on Antennas and Propagation, vol. 41, no. 10, Oct 1993,
 pp. 1448--1455

 Date: 06.20.2019
 
 Paul Zhang (pzpzpzp1@mit.edu)
 *
 * This is Paul's c++ implementation of the paper above based primarily on Fabio Freschi's (fabio.freschi@polito.it) matlab toolbox.
 * https://www.mathworks.com/matlabcentral/fileexchange/47782-int_green3d
 * I make no claims that this is efficient code. It also won't be maintained/supported. 
 * This is literally line by line naive translation of Fabio's matlab implementation.
 *
 * To use it just include it into your code like you would any header file. This code depends on Eigen for convenience.
 */

#include <cstdlib>
#include <vector>
#include "Eigen/Dense"
#include "Eigen/Core"
#include <Eigen/Geometry>
#include <iostream>
#include <fstream>
#include <math.h>
#include <map>

Eigen::Vector3f IG3D_eulerAngleFunc(Eigen::Vector3f P0, Eigen::Vector3f Pz, Eigen::Vector3f Px) {
    Eigen::Vector3f Uz = Pz-P0;
    Eigen::Vector3f Ux = Px-P0;

    Uz.normalize();
    Ux.normalize();

    auto phi = std::atan2(Uz(0),-Uz(1));
    
    auto eta = -Uz(0)*std::sin(phi)+Uz(1)*std::cos(phi);
    auto theta = std::acos(Uz(2));
    
    auto ineg = eta>0;
    theta = ineg ? -theta : theta;
    Eigen::Vector3f Uxr; Uxr << std::cos(phi), std::sin(phi), 0;
    eta = -Ux(0)*cos(theta)*sin(phi) + Ux(1)*cos(theta)*cos(phi) + Ux(2)*sin(theta);
    auto cosarg = Ux.dot(Uxr);
    cosarg = cosarg > 1 ? 1 : cosarg;
    cosarg = cosarg < -1 ? -1 : cosarg;

    double psi = std::acos(cosarg);
    
    ineg = (eta < 0);
    psi = ineg ? -psi : psi;
    Eigen::Vector3f toreturn; toreturn << phi, theta, psi;
    return toreturn;
}

Eigen::Matrix3f IG3D_rotation(double phi, double theta, double psi)
{   
    auto cosphi = std::cos(phi);
    auto sinphi = std::sin(phi);
    auto costheta = std::cos(theta);
    auto sintheta = std::sin(theta);
    auto cospsi = std::cos(psi);
    auto sinpsi = std::sin(psi);

    //std::cout << cosphi << " " << sinphi << " " << costheta << " " << sintheta << " " << cospsi << " " << sinpsi << "\n"; 
    
    Eigen::Matrix3f R;
    R(0,0) = cospsi*cosphi-costheta*sinphi*sinpsi;
    R(0,1) = -sinpsi*cosphi-costheta*sinphi*cospsi;
    R(0,2) = sintheta*sinphi;

    R(1,0) = cospsi*sinphi+costheta*cosphi*sinpsi;
    R(1,1) = -sinpsi*sinphi+costheta*cosphi*cospsi;
    R(1,2) = -sintheta*cosphi;
    
    R(2,0) = sinpsi*sintheta;
    R(2,1) = cospsi*sintheta;
    R(2,2) = costheta;
    return R;
}

std::vector<std::pair<double, Eigen::Vector3f>> computeGreensIntegral3D(std::vector<Eigen::Vector3f> Pglo, std::vector<Eigen::Vector3f> V){
    
    int nfp = Pglo.size();
    Eigen::Vector3f P0 = V[0];
    Eigen::Vector3f Px = V[1];
    Eigen::Vector3f v1 = V[1] - V[0];
    Eigen::Vector3f v2 = V[2] - V[0];

    Eigen::Vector3f dz; dz << v1(1)*v2(2)-v2(1)*v1(2), v2(0)*v1(2)-v1(0)*v2(2), v1(0)*v2(1)-v2(0)*v1(1); dz.normalize();
    Eigen::Vector3f Pz = P0 + dz;
    
    auto eas = IG3D_eulerAngleFunc(P0, Pz, Px);

    double phi = eas(0);
    double theta = eas(1);
    double psi = eas(2);
    
    //std::cout << "phi theta psi: " << phi << " " << theta << " "<< psi <<"\n";
    
    auto R = IG3D_rotation(phi,theta,psi);
    //std::cout << "R: " << R <<"\n";
    
    std::vector<Eigen::Vector3f> Ploc(nfp);
    for(int i =0; i < nfp; i++){
        Ploc[i] = R.transpose()*(Pglo[i] - V[0]);
    }
    
    Eigen::VectorXf u0(nfp);
    Eigen::VectorXf v0(nfp);
    Eigen::VectorXf w0(nfp);
    for(int i=0;i<nfp;i++){
        u0(i) = Ploc[i](0);
        v0(i) = Ploc[i](1);
        w0(i) = Ploc[i](2);
    }

    std::vector<Eigen::Vector3f> Vloc(3);
    for(int i =0; i < 3; i++){
        Vloc[i] = R.transpose()*(V[i] - V[0]);
    }

    double l3 = Vloc[1](0);
    double u3 = Vloc[2](0);
    double v3 = Vloc[2](1);

    double l1 = std::sqrt((l3-u3)*(l3-u3) + v3*v3);
    double l2 = std::sqrt(u3*u3 + v3*v3);

    double threshold = l1 < l2 ? l1 : l2;
    threshold = threshold < l3 ? threshold : l3;
    threshold*=1e-6;
    
    for(int i=0;i<nfp;i++){
        w0[i] = std::abs(w0[i])<threshold ? 0 : w0[i];
    }
    
    Eigen::MatrixXf w0rep(nfp,3);
    for(int i=0;i<nfp;i++){
        w0rep(i,0)=w0[i];
        w0rep(i,1)=w0[i];
        w0rep(i,2)=w0[i];
    }

    Eigen::Vector3f mv1,mv2,mv3;
    mv1 << v3, l3-u3, 0;
    mv2 << -v3, u3, 0;
    mv3 << 0, -l3, 0;
    mv1.normalize();
    mv2.normalize();
    mv3.normalize();
    Eigen::Matrix3f m;
    m << mv1(0),mv1(1),mv1(2),
        mv2(0),mv2(1),mv2(2),
        mv3(0),mv3(1),mv3(2);
 
    Eigen::MatrixXf sminus(nfp,3);
    for(int i=0; i<nfp; i++){
        sminus(i,0) = -((l3-u3)*(-u0(i) + l3) + v3*v0(i))/l1; 
        sminus(i,1) = -(u3*(u3-u0(i))+v3*(v3-v0(i)))/l2; 
        sminus(i,2) = -u0(i); 
    }
    
    Eigen::MatrixXf splus(nfp,3);
    for(int i=0; i<nfp; i++){
        splus(i,0) = ((u3-l3)*(u3-u0(i))+v3*(v3-v0(i)))/l1;
        splus(i,1) = (u3*u0(i)+v3*v0(i))/l2;
        splus(i,2) = l3-u0(i);
    }
    
    Eigen::MatrixXf t0(nfp,3);
    for(int i=0; i<nfp; i++){
        t0(i,0) = ((u3-l3)*v0(i)+v3*(l3-u0(i)))/l1;
        t0(i,1) = (v3*u0(i)-u3*v0(i))/l2;
        t0(i,2) = v0(i);
    }
    
    Eigen::MatrixXf tplus(nfp,3);
    for(int i=0; i<nfp; i++){
        auto tplus1a = u3-u0(i);
        auto tplus1b = v3-v0(i);
        auto tplus2a = u0(i);
        auto tplus2b = v0(i);
        auto tplus3a = l3-u0(i);
        auto tplus3b = v0(i);
        
        tplus(i,0) = std::sqrt(tplus1a*tplus1a + tplus1b*tplus1b);
        tplus(i,1) = std::sqrt(tplus2a*tplus2a + tplus2b*tplus2b);
        tplus(i,2) = std::sqrt(tplus3a*tplus3a + tplus3b*tplus3b);
    }
    
    Eigen::MatrixXf tminus(nfp,3);
    for(int i=0; i<nfp; i++){
        tminus(i,0) = tplus(i,2);
        tminus(i,1) = tplus(i,0);
        tminus(i,2) = tplus(i,1);
    }

    Eigen::MatrixXf R0(nfp,3);
    for(int i=0; i<nfp; i++){
        R0(i,0) = std::sqrt(t0(i,0)*t0(i,0) + w0rep(i,0)*w0rep(i,0));
        R0(i,1) = std::sqrt(t0(i,1)*t0(i,1) + w0rep(i,1)*w0rep(i,1));
        R0(i,2) = std::sqrt(t0(i,2)*t0(i,2) + w0rep(i,2)*w0rep(i,2));
    }
    
    Eigen::MatrixXf Rminus(nfp,3);
    for(int i=0; i<nfp; i++){
        Rminus(i,0) = std::sqrt(tminus(i,0)*tminus(i,0) + w0rep(i,0)*w0rep(i,0));
        Rminus(i,1) = std::sqrt(tminus(i,1)*tminus(i,1) + w0rep(i,1)*w0rep(i,1));
        Rminus(i,2) = std::sqrt(tminus(i,2)*tminus(i,2) + w0rep(i,2)*w0rep(i,2));
    }
    
    Eigen::MatrixXf Rplus(nfp,3);
    for(int i=0; i<nfp; i++){
        Rplus(i,0) = std::sqrt(tplus(i,0)*tplus(i,0) + w0rep(i,0)*w0rep(i,0));
        Rplus(i,1) = std::sqrt(tplus(i,1)*tplus(i,1) + w0rep(i,1)*w0rep(i,1));
        Rplus(i,2) = std::sqrt(tplus(i,2)*tplus(i,2) + w0rep(i,2)*w0rep(i,2));
    }

    Eigen::MatrixXf f2(nfp,3);
    Eigen::MatrixXf beta(nfp,3);
    
    for(int i=0; i<nfp; i++){
        for(int j=0; j<3; j++){
            //id1 condition
            if(std::abs(w0rep(i,j))  >= threshold){
                f2(i,j) = std::log((Rplus(i,j)+splus(i,j))/(Rminus(i,j)+sminus(i,j)));
                beta(i,j) = std::atan(t0(i,j)*splus(i,j)/(R0(i,j)*R0(i,j) + std::abs(w0rep(i,j))*Rplus(i,j))) - std::atan(t0(i,j)*sminus(i,j)/(R0(i,j)*R0(i,j) + std::abs(w0rep(i,j))*Rminus(i,j)));
            }
                    
            //id2 condition
            if((std::abs(w0rep(i,j)) < threshold) & (std::abs(t0(i,j)) >= threshold)) {
                f2(i,j) = std::log((tplus(i,j) + splus(i,j)) / (tminus(i,j) + sminus(i,j)));                 // % eq. (15)
                beta(i,j) = std::atan(splus(i,j)/t0(i,j)) - std::atan(sminus(i,j) / t0(i,j));                 //  % eq. (17)
            }
            
            //id3 condition
            if((std::abs(w0rep(i,j)) < threshold) & (std::abs(t0(i,j)) < threshold)){
                beta(i,j) = 0;
                f2(i,j) = std::abs(std::log(splus(i,j)/sminus(i,j)));               //   % abs(lim t->0 eq. 15)
            }
            
            // f2 finite
            if(!std::isfinite(f2(i,j)))
            {
                f2(i,j) = 0;
            }
        }
    }
    
    std::vector<std::pair<double, Eigen::Vector3f>> toReturn(nfp);
    for(int i=0;i<nfp;i++){
        double I1 = 0;
        for(int j=0;j<3;j++){
            I1+= t0(i,j)*f2(i,j) - std::abs(w0rep(i,j))*beta(i,j);
        }
        
        auto w0sign = (w0[i] > 0) ? 1 : -1;
        Eigen::Vector3f Igradloc;
        Igradloc(0) = -m(0,0)*f2(i,0)-m(1,0)*f2(i,1)-m(2,0)*f2(i,2);
        Igradloc(1) = -m(0,1)*f2(i,0)-m(1,1)*f2(i,1)-m(2,1)*f2(i,2);
        Igradloc(2) = -w0sign * (beta(i,0)+beta(i,1)+beta(i,2));
        
        auto Igrad = R * Igradloc;
        toReturn[i]=std::pair<double, Eigen::Vector3f>(I1, Igrad);
    }

    return toReturn;
}
