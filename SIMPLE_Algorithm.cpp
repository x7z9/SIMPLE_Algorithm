/*
 * SIMPLE Algorithm Implementation in C++
 * Solves 2D Lid-Driven Cavity Problem
 * Author: J. Yadagiri | Civil Engineer specializing in CFD
 * https://www.linkedin.com/in/j-yadagiri-4944ba21b?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app
 * Semi-Implicit Method for Pressure Linked Equations (SIMPLE)
 * Reference: Patankar & Spalding (1972)
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

class SIMPLESolver {
private:
    int nx, ny;                    // Grid dimensions
    double dx, dy;                 // Grid spacing
    double rho, mu;                // Density and dynamic viscosity
    double nu;                     // Kinematic viscosity
    double dt;                     // Time step
    double alpha_p, alpha_u, alpha_v;  // Under-relaxation factors
    
    // Field variables
    std::vector<std::vector<double>> u, v, p;           // Velocity and pressure
    std::vector<std::vector<double>> u_star, v_star;    // Intermediate velocities
    std::vector<std::vector<double>> p_prime;           // Pressure correction
    
public:
    SIMPLESolver(int nx_, int ny_, double L, double rho_, double mu_) 
        : nx(nx_), ny(ny_), rho(rho_), mu(mu_) {
        
        dx = L / (nx - 1);
        dy = L / (ny - 1);
        nu = mu / rho;
        dt = 0.001;
        
        // Under-relaxation factors (critical for stability)
        alpha_u = 0.7;
        alpha_v = 0.7;
        alpha_p = 0.3;
        
        // Initialize grids
        u.resize(nx, std::vector<double>(ny, 0.0));
        v.resize(nx, std::vector<double>(ny, 0.0));
        p.resize(nx, std::vector<double>(ny, 0.0));
        u_star.resize(nx, std::vector<double>(ny, 0.0));
        v_star.resize(nx, std::vector<double>(ny, 0.0));
        p_prime.resize(nx, std::vector<double>(ny, 0.0));
    }
    
    // Solve momentum equations (predictor step)
    void solveMomentumEquations() {
        // X-momentum equation discretization
        for (int i = 1; i < nx-1; i++) {
            for (int j = 1; j < ny-1; j++) {
                double u_e = 0.5 * (u[i][j] + u[i+1][j]);
                double u_w = 0.5 * (u[i][j] + u[i-1][j]);
                double u_n = 0.5 * (u[i][j] + u[i][j+1]);
                double u_s = 0.5 * (u[i][j] + u[i][j-1]);
                
                // Convective terms
                double conv_x = (u_e * u_e - u_w * u_w) / dx;
                double conv_y = (u_n * v[i][j+1] - u_s * v[i][j-1]) / dy;
                
                // Diffusive terms (Laplacian)
                double diff_x = nu * (u[i+1][j] - 2*u[i][j] + u[i-1][j]) / (dx*dx);
                double diff_y = nu * (u[i][j+1] - 2*u[i][j] + u[i][j-1]) / (dy*dy);
                
                // Pressure gradient
                double dp_dx = (p[i+1][j] - p[i-1][j]) / (2*dx);
                
                // Update intermediate velocity u*
                u_star[i][j] = u[i][j] + dt * (-conv_x - conv_y + diff_x + diff_y - dp_dx/rho);
            }
        }
        
        // Y-momentum equation discretization
        for (int i = 1; i < nx-1; i++) {
            for (int j = 1; j < ny-1; j++) {
                double v_e = 0.5 * (v[i][j] + v[i+1][j]);
                double v_w = 0.5 * (v[i][j] + v[i-1][j]);
                double v_n = 0.5 * (v[i][j] + v[i][j+1]);
                double v_s = 0.5 * (v[i][j] + v[i][j-1]);
                
                // Convective terms
                double conv_x = (u[i+1][j] * v_e - u[i-1][j] * v_w) / dx;
                double conv_y = (v_n * v_n - v_s * v_s) / dy;
                
                // Diffusive terms
                double diff_x = nu * (v[i+1][j] - 2*v[i][j] + v[i-1][j]) / (dx*dx);
                double diff_y = nu * (v[i][j+1] - 2*v[i][j] + v[i][j-1]) / (dy*dy);
                
                // Pressure gradient
                double dp_dy = (p[i][j+1] - p[i][j-1]) / (2*dy);
                
                // Update intermediate velocity v*
                v_star[i][j] = v[i][j] + dt * (-conv_x - conv_y + diff_x + diff_y - dp_dy/rho);
            }
        }
    }
    
    // Solve pressure correction equation (Poisson equation)
    void solvePressureCorrection(int max_iter = 100) {
        for (int iter = 0; iter < max_iter; iter++) {
            double max_residual = 0.0;
            
            for (int i = 1; i < nx-1; i++) {
                for (int j = 1; j < ny-1; j++) {
                    // Continuity equation residual (mass source)
                    double div_u = (u_star[i+1][j] - u_star[i-1][j]) / (2*dx) +
                                   (v_star[i][j+1] - v_star[i][j-1]) / (2*dy);
                    
                    // Pressure correction Poisson equation
                    double a_p = 2.0/(dx*dx) + 2.0/(dy*dy);
                    double a_e = 1.0/(dx*dx);
                    double a_w = 1.0/(dx*dx);
                    double a_n = 1.0/(dy*dy);
                    double a_s = 1.0/(dy*dy);
                    
                    double p_prime_new = (1.0/a_p) * (
                        a_e * p_prime[i+1][j] + a_w * p_prime[i-1][j] +
                        a_n * p_prime[i][j+1] + a_s * p_prime[i][j-1] -
                        rho * div_u / dt
                    );
                    
                    double residual = std::abs(p_prime_new - p_prime[i][j]);
                    max_residual = std::max(max_residual, residual);
                    
                    p_prime[i][j] = p_prime_new;
                }
            }
            
            if (max_residual < 1e-6) break;
        }
    }
    
    // Correct velocities and pressure
    void correctVelocitiesAndPressure() {
        for (int i = 1; i < nx-1; i++) {
            for (int j = 1; j < ny-1; j++) {
                // Velocity correction
                double dp_prime_dx = (p_prime[i+1][j] - p_prime[i-1][j]) / (2*dx);
                double dp_prime_dy = (p_prime[i][j+1] - p_prime[i][j-1]) / (2*dy);
                
                u[i][j] = u_star[i][j] - (dt/rho) * dp_prime_dx;
                v[i][j] = v_star[i][j] - (dt/rho) * dp_prime_dy;
                
                // Pressure correction with under-relaxation
                p[i][j] += alpha_p * p_prime[i][j];
            }
        }
        
        // Apply under-relaxation to velocities
        for (int i = 1; i < nx-1; i++) {
            for (int j = 1; j < ny-1; j++) {
                u[i][j] = alpha_u * u[i][j] + (1 - alpha_u) * u_star[i][j];
                v[i][j] = alpha_v * v[i][j] + (1 - alpha_v) * v_star[i][j];
            }
        }
    }
    
    // Apply boundary conditions (Lid-Driven Cavity)
    void applyBoundaryConditions() {
        // Top wall (moving lid)
        for (int i = 0; i < nx; i++) {
            u[i][ny-1] = 1.0;  // Lid velocity
            v[i][ny-1] = 0.0;
            p[i][ny-1] = p[i][ny-2];
        }
        
        // Bottom, left, right walls (no-slip)
        for (int i = 0; i < nx; i++) {
            u[i][0] = 0.0;
            v[i][0] = 0.0;
            p[i][0] = p[i][1];
        }
        
        for (int j = 0; j < ny; j++) {
            u[0][j] = 0.0;
            v[0][j] = 0.0;
            p[0][j] = p[1][j];
            
            u[nx-1][j] = 0.0;
            v[nx-1][j] = 0.0;
            p[nx-1][j] = p[nx-2][j];
        }
    }
    
    // Main SIMPLE iteration
    void solve(int max_iterations, double convergence_criterion) {
        std::cout << "Starting SIMPLE Algorithm...\n";
        std::cout << "Grid: " << nx << "x" << ny << "\n";
        std::cout << "Reynolds Number: " << (1.0 * 1.0 / nu) << "\n\n";
        
        for (int iter = 0; iter < max_iterations; iter++) {
            // Step 1: Solve momentum equations
            solveMomentumEquations();
            
            // Step 2: Solve pressure correction equation
            solvePressureCorrection(50);
            
            // Step 3: Correct velocities and pressure
            correctVelocitiesAndPressure();
            
            // Step 4: Apply boundary conditions
            applyBoundaryConditions();
            
            // Check convergence
            if (iter % 100 == 0) {
                double max_div = computeMaxDivergence();
                std::cout << "Iteration " << std::setw(5) << iter 
                          << " | Max Divergence: " << std::scientific 
                          << max_div << std::endl;
                
                if (max_div < convergence_criterion) {
                    std::cout << "\nConverged in " << iter << " iterations!\n";
                    break;
                }
            }
        }
    }
    
    // Compute maximum divergence for convergence check
    double computeMaxDivergence() {
        double max_div = 0.0;
        for (int i = 1; i < nx-1; i++) {
            for (int j = 1; j < ny-1; j++) {
                double div = std::abs((u[i+1][j] - u[i-1][j]) / (2*dx) +
                                     (v[i][j+1] - v[i][j-1]) / (2*dy));
                max_div = std::max(max_div, div);
            }
        }
        return max_div;
    }
    
    // Export results to CSV for visualization
    void exportResults(const std::string& filename) {
        std::ofstream file(filename);
        file << "x,y,u,v,p\n";
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                file << i*dx << "," << j*dy << "," 
                     << u[i][j] << "," << v[i][j] << "," 
                     << p[i][j] << "\n";
            }
        }
        file.close();
        std::cout << "Results exported to " << filename << "\n";
    }
};

// Main function
int main() {
    // Problem setup
    int nx = 64;          // Grid points in x
    int ny = 64;          // Grid points in y
    double L = 1.0;       // Domain size
    double rho = 1.0;     // Density
    double mu = 0.01;     // Dynamic viscosity (Re = 100)
    
    // Create solver
    SIMPLESolver solver(nx, ny, L, rho, mu);
    
    // Solve
    solver.solve(5000, 1e-6);
    
    // Export results
    solver.exportResults("simple_results.csv");
    
    return 0;
}
