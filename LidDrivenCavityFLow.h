#pragma once
#include "Data.h"

class LidDrivenCavityFLow {
	std::string grid;
	double Re;
	int Nx;
	int Ny;
	double dx;
	double dy;
	double alpha_p;
	double alpha_u;
	double alpha_v;
	double tol;
	void initialize(Data& data) {
		if (grid == "staggered") {
			data.x_p = arma::linspace(0.5 * dx, 1.0 - 0.5 * dx, Nx);
			data.y_p = arma::linspace(0.5 * dy, 1.0 - 0.5 * dy, Ny);
			data.x_u = arma::linspace(0.0, 1.0, Nx + 1);
			data.y_u = arma::linspace(0.5 * dy, 1.0 - 0.5 * dy, Ny);
			data.x_v = arma::linspace(0.5 * dx, 1.0 - 0.5 * dx, Nx);
			data.y_v = arma::linspace(0.0, 1.0, Ny + 1);
			data.p = arma::zeros(Nx, Ny);
			data.u = arma::zeros(Nx + 1, Ny);
			data.v = arma::zeros(Nx, Ny + 1);
		}
		else if (grid == "collocated") {
			data.x_p = arma::linspace(0.5 * dx, 1.0 - 0.5 * dx, Nx);
			data.y_p = arma::linspace(0.5 * dy, 1.0 - 0.5 * dy, Ny);
			data.x_u = arma::linspace(0.5 * dx, 1.0 - 0.5 * dx, Nx);
			data.y_u = arma::linspace(0.5 * dy, 1.0 - 0.5 * dy, Ny);
			data.x_v = arma::linspace(0.5 * dx, 1.0 - 0.5 * dx, Nx);
			data.y_v = arma::linspace(0.5 * dy, 1.0 - 0.5 * dy, Ny);
			data.p = arma::zeros(Nx, Ny);
			data.u = arma::zeros(Nx, Ny);
			data.v = arma::zeros(Nx, Ny);
		}
	}
	void SIMPLE(Data& data) {
		int count = 0;
		if (grid == "staggered") {
			while (true) {
				count++;
				arma::mat u_star = arma::zeros(Nx + 1, Ny);
				arma::mat A_W_u = arma::zeros(Nx + 1, Ny);
				arma::mat A_S_u = arma::zeros(Nx + 1, Ny);
				arma::mat A_P_u = arma::zeros(Nx + 1, Ny);
				arma::mat A_N_u = arma::zeros(Nx + 1, Ny);
				arma::mat A_E_u = arma::zeros(Nx + 1, Ny);
				arma::mat Q_P_u = arma::zeros(Nx + 1, Ny);
				for (int i = 0; i < Nx + 1; i++) {
					for (int j = 0; j < Ny; j++) {
						if (i == 0 || i == Nx) {
							A_P_u(i, j) = 1.0;
						}
						else {
							A_P_u(i, j) += (data.u(i, j) + data.u(i + 1, j)) / 4 * dy;
							A_E_u(i, j) += (data.u(i, j) + data.u(i + 1, j)) / 4 * dy;
							A_W_u(i, j) += -(data.u(i - 1, j) + data.u(i, j)) / 4 * dy;
							A_P_u(i, j) += -(data.u(i - 1, j) + data.u(i, j)) / 4 * dy;
							if (j == Ny - 1) {
								Q_P_u(i, j) += -(data.v(i - 1, j + 1) + data.v(i, j + 1)) / 2 * dx;
							}
							else {
								A_P_u(i, j) += (data.v(i - 1, j + 1) + data.v(i, j + 1)) / 4 * dx;
								A_N_u(i, j) += (data.v(i - 1, j + 1) + data.v(i, j + 1)) / 4 * dx;
							}
							if (j > 0) {
								A_S_u(i, j) += -(data.v(i - 1, j) + data.v(i, j)) / 4 * dx;
								A_P_u(i, j) += -(data.v(i - 1, j) + data.v(i, j)) / 4 * dx;
							}
							A_P_u(i, j) += dy / (Re * dx);
							A_E_u(i, j) += -dy / (Re * dx);
							A_W_u(i, j) += -dy / (Re * dx);
							A_P_u(i, j) += dy / (Re * dx);
							if (j == Ny - 1) {
								A_P_u(i, j) += 2 * dx / (Re * dy);
								Q_P_u(i, j) += 2 * dx / (Re * dy);
							}
							else {
								A_P_u(i, j) += dx / (Re * dy);
								A_N_u(i, j) += -dx / (Re * dy);
							}
							if (j == 0) {
								A_P_u(i, j) += 2 * dx / (Re * dy);
							}
							else {
								A_S_u(i, j) += -dx / (Re * dy);
								A_P_u(i, j) += dx / (Re * dy);
							}
							Q_P_u(i, j) += (data.p(i - 1, j) - data.p(i, j)) * dy;
						}
					}
				}
				arma::mat A_u;
				arma::vec b_u;
				assemble(A_u, b_u, A_W_u, A_S_u, A_P_u, A_N_u, A_E_u, Q_P_u);
				b_u = arma::solve(A_u, b_u);
				reshape(u_star, b_u);
				arma::mat v_star = arma::zeros(Nx, Ny + 1);
				arma::mat A_W_v = arma::zeros(Nx, Ny + 1);
				arma::mat A_S_v = arma::zeros(Nx, Ny + 1);
				arma::mat A_P_v = arma::zeros(Nx, Ny + 1);
				arma::mat A_N_v = arma::zeros(Nx, Ny + 1);
				arma::mat A_E_v = arma::zeros(Nx, Ny + 1);
				arma::mat Q_P_v = arma::zeros(Nx, Ny + 1);
				for (int i = 0; i < Nx; i++) {
					for (int j = 0; j < Ny + 1; j++) {
						if (j == 0 || j == Ny) {
							A_P_v(i, j) = 1.0;
						}
						else {
							if (i < Nx - 1) {
								A_P_v(i, j) += (data.u(i + 1, j - 1) + data.u(i + 1, j)) / 4 * dy;
								A_E_v(i, j) += (data.u(i + 1, j - 1) + data.u(i + 1, j)) / 4 * dy;
							}
							if (i > 0) {
								A_W_v(i, j) += -(data.u(i, j - 1) + data.u(i, j)) / 4 * dy;
								A_P_v(i, j) += -(data.u(i, j - 1) + data.u(i, j)) / 4 * dy;
							}
							A_P_v(i, j) += (data.v(i, j) + data.v(i, j + 1)) / 4 * dx;
							A_N_v(i, j) += (data.v(i, j) + data.v(i, j + 1)) / 4 * dx;
							A_S_v(i, j) += -(data.v(i, j - 1) + data.v(i, j)) / 4 * dx;
							A_P_v(i, j) += -(data.v(i, j - 1) + data.v(i, j)) / 4 * dx;
							if (i == Nx - 1) {
								A_P_v(i, j) += 2 * dy / (Re * dx);
							}
							else {
								A_P_v(i, j) += dy / (Re * dx);
								A_E_v(i, j) += -dy / (Re * dx);
							}
							if (i == 0) {
								A_P_v(i, j) += 2 * dy / (Re * dx);
							}
							else {
								A_W_v(i, j) += -dy / (Re * dx);
								A_P_v(i, j) += dy / (Re * dx);
							}
							A_P_v(i, j) += dx / (Re * dy);
							A_N_v(i, j) += -dx / (Re * dy);
							A_S_v(i, j) += -dx / (Re * dy);
							A_P_v(i, j) += dx / (Re * dy);
							Q_P_v(i, j) += (data.p(i, j - 1) - data.p(i, j)) * dx;
						}
					}
				}
				arma::mat A_v;
				arma::vec b_v;
				assemble(A_v, b_v, A_W_v, A_S_v, A_P_v, A_N_v, A_E_v, Q_P_v);
				b_v = arma::solve(A_v, b_v);
				reshape(v_star, b_v);
				arma::mat p_prime = arma::zeros(Nx, Ny);
				arma::mat A_W_p = arma::zeros(Nx, Ny);
				arma::mat A_S_p = arma::zeros(Nx, Ny);
				arma::mat A_P_p = arma::zeros(Nx, Ny);
				arma::mat A_N_p = arma::zeros(Nx, Ny);
				arma::mat A_E_p = arma::zeros(Nx, Ny);
				arma::mat Q_P_p = arma::zeros(Nx, Ny);
				for (int i = 0; i < Nx; i++) {
					for (int j = 0; j < Ny; j++) {
						if (i == 0 && j == 0) {
							A_P_p(i, j) = 1.0;
						}
						else {
							if (i < Nx - 1) {
								A_P_p(i, j) += pow(dy, 2) / A_P_u(i + 1, j);
								A_E_p(i, j) += -pow(dy, 2) / A_P_u(i + 1, j);
							}
							Q_P_p(i, j) += -u_star(i + 1, j) * dy;
							if (i > 0) {
								A_W_p(i, j) += -pow(dy, 2) / A_P_u(i, j);
								A_P_p(i, j) += pow(dy, 2) / A_P_u(i, j);
							}
							Q_P_p(i, j) += u_star(i, j) * dy;
							if (j < Ny - 1) {
								A_P_p(i, j) += pow(dx, 2) / A_P_v(i, j + 1);
								A_N_p(i, j) += -pow(dx, 2) / A_P_v(i, j + 1);
							}
							Q_P_p(i, j) += -v_star(i, j + 1) * dx;
							if (j > 0) {
								A_S_p(i, j) += -pow(dx, 2) / A_P_v(i, j);
								A_P_p(i, j) += pow(dx, 2) / A_P_v(i, j);
							}
							Q_P_p(i, j) += v_star(i, j) * dx;
						}
					}
				}
				arma::mat A_p;
				arma::vec b_p;
				assemble(A_p, b_p, A_W_p, A_S_p, A_P_p, A_N_p, A_E_p, Q_P_p);
				b_p = arma::solve(A_p, b_p);
				reshape(p_prime, b_p);
				arma::mat u_prime = arma::zeros(Nx + 1, Ny);
				for (int i = 1; i < Nx; i++) {
					for (int j = 0; j < Ny; j++) {
						u_prime(i, j) = dy / A_P_u(i, j) * (p_prime(i - 1, j) - p_prime(i, j));
					}
				}
				arma::mat v_prime = arma::zeros(Nx, Ny + 1);
				for (int i = 0; i < Nx; i++) {
					for (int j = 1; j < Ny; j++) {
						v_prime(i, j) = dx / A_P_v(i, j) * (p_prime(i, j - 1) - p_prime(i, j));
					}
				}
				arma::mat u = u_star + u_prime;
				arma::mat v = v_star + v_prime;
				arma::mat p_new = data.p + alpha_p * p_prime;
				arma::mat u_new = alpha_u * u + (1 - alpha_u) * data.u;
				arma::mat v_new = alpha_v * v + (1 - alpha_v) * data.v;
				double res_p = arma::abs(data.p - p_new).max();
				double res_u = arma::abs(data.u - u_new).max();
				double res_v = arma::abs(data.v - v_new).max();
				std::cout << "grid=" << grid << " Re=" << Re << " count=" << count << " res_p=" << res_p << " res_u=" << res_u << " res_v=" << res_v << std::endl;
				data.p = p_new;
				data.u = u_new;
				data.v = v_new;
				if (res_p < tol && res_u < tol && res_v < tol) {
					break;
				}
			}
		}
		else if (grid == "collocated") {
			arma::mat u_f_old = arma::zeros(Nx + 1, Ny);
			for (int i = 1; i < Nx; i++) {
				for (int j = 0; j < Ny; j++) {
					u_f_old(i, j) = (data.u(i - 1, j) + data.u(i, j)) / 2;
				}
			}
			arma::mat v_f_old = arma::zeros(Nx, Ny + 1);
			for (int i = 0; i < Nx; i++) {
				for (int j = 1; j < Ny; j++) {
					v_f_old(i, j) = (data.v(i, j - 1) + data.v(i, j)) / 2;
				}
			}
			while (true) {
				count++;
				arma::mat u_star = arma::zeros(Nx, Ny);
				arma::mat A_W_u = arma::zeros(Nx, Ny);
				arma::mat A_S_u = arma::zeros(Nx, Ny);
				arma::mat A_P_u = arma::zeros(Nx, Ny);
				arma::mat A_N_u = arma::zeros(Nx, Ny);
				arma::mat A_E_u = arma::zeros(Nx, Ny);
				arma::mat Q_P_u = arma::zeros(Nx, Ny);
				for (int i = 0; i < Nx; i++) {
					for (int j = 0; j < Ny; j++) {
						if (i < Nx - 1) {
							A_P_u(i, j) += u_f_old(i + 1, j) / 2 * dy;
							A_E_u(i, j) += u_f_old(i + 1, j) / 2 * dy;
						}
						if (i > 0) {
							A_W_u(i, j) += -u_f_old(i, j) / 2 * dy;
							A_P_u(i, j) += -u_f_old(i, j) / 2 * dy;
						}
						if (j == Ny - 1) {
							Q_P_u(i, j) += -v_f_old(i, j + 1) * dx;
						}
						else {
							A_P_u(i, j) += v_f_old(i, j + 1) / 2 * dx;
							A_N_u(i, j) += v_f_old(i, j + 1) / 2 * dx;
						}
						if (j > 0) {
							A_S_u(i, j) += -v_f_old(i, j) / 2 * dx;
							A_P_u(i, j) += -v_f_old(i, j) / 2 * dx;
						}
						if (i == Nx - 1) {
							A_P_u(i, j) += 2 * dy / (Re * dx);
						}
						else {
							A_P_u(i, j) += dy / (Re * dx);
							A_E_u(i, j) += -dy / (Re * dx);
						}
						if (i == 0) {
							A_P_u(i, j) += 2 * dy / (Re * dx);
						}
						else {
							A_W_u(i, j) += -dy / (Re * dx);
							A_P_u(i, j) += dy / (Re * dx);
						}
						if (j == Ny - 1) {
							A_P_u(i, j) += 2 * dx / (Re * dy);
							Q_P_u(i, j) += 2 * dx / (Re * dy);
						}
						else {
							A_P_u(i, j) += dx / (Re * dy);
							A_N_u(i, j) += -dx / (Re * dy);
						}
						if (j == 0) {
							A_P_u(i, j) += 2 * dx / (Re * dy);
						}
						else {
							A_S_u(i, j) += -dx / (Re * dy);
							A_P_u(i, j) += dx / (Re * dy);
						}
						if (i == 0) {
							Q_P_u(i, j) += (data.p(i, j) - data.p(i + 1, j)) / 2 * dy;
						}
						else if (i == Nx - 1) {
							Q_P_u(i, j) += (data.p(i - 1, j) - data.p(i, j)) / 2 * dy;
						}
						else {
							Q_P_u(i, j) += (data.p(i - 1, j) - data.p(i + 1, j)) / 2 * dy;
						}
					}
				}
				arma::mat A_u;
				arma::vec b_u;
				assemble(A_u, b_u, A_W_u, A_S_u, A_P_u, A_N_u, A_E_u, Q_P_u);
				b_u = arma::solve(A_u, b_u);
				reshape(u_star, b_u);
				arma::mat v_star = arma::zeros(Nx, Ny);
				arma::mat A_W_v = arma::zeros(Nx, Ny);
				arma::mat A_S_v = arma::zeros(Nx, Ny);
				arma::mat A_P_v = arma::zeros(Nx, Ny);
				arma::mat A_N_v = arma::zeros(Nx, Ny);
				arma::mat A_E_v = arma::zeros(Nx, Ny);
				arma::mat Q_P_v = arma::zeros(Nx, Ny);
				for (int i = 0; i < Nx; i++) {
					for (int j = 0; j < Ny; j++) {
						if (i < Nx - 1) {
							A_P_v(i, j) += u_f_old(i + 1, j) / 2 * dy;
							A_E_v(i, j) += u_f_old(i + 1, j) / 2 * dy;
						}
						if (i > 0) {
							A_W_v(i, j) += -u_f_old(i, j) / 2 * dy;
							A_P_v(i, j) += -u_f_old(i, j) / 2 * dy;
						}
						if (j < Ny - 1) {
							A_P_v(i, j) += v_f_old(i, j + 1) / 2 * dx;
							A_N_v(i, j) += v_f_old(i, j + 1) / 2 * dx;
						}
						if (j > 0) {
							A_S_v(i, j) += -v_f_old(i, j) / 2 * dx;
							A_P_v(i, j) += -v_f_old(i, j) / 2 * dx;
						}
						if (i == Nx - 1) {
							A_P_v(i, j) += 2 * dy / (Re * dx);
						}
						else {
							A_P_v(i, j) += dy / (Re * dx);
							A_E_v(i, j) += -dy / (Re * dx);
						}
						if (i == 0) {
							A_P_v(i, j) += 2 * dy / (Re * dx);
						}
						else {
							A_W_v(i, j) += -dy / (Re * dx);
							A_P_v(i, j) += dy / (Re * dx);
						}
						if (j == Ny - 1) {
							A_P_v(i, j) += 2 * dx / (Re * dy);
						}
						else {
							A_P_v(i, j) += dx / (Re * dy);
							A_N_v(i, j) += -dx / (Re * dy);
						}
						if (j == 0) {
							A_P_v(i, j) += 2 * dx / (Re * dy);
						}
						else {
							A_S_v(i, j) += -dx / (Re * dy);
							A_P_v(i, j) += dx / (Re * dy);
						}
						if (j == 0) {
							Q_P_v(i, j) += (data.p(i, j) - data.p(i, j + 1)) / 2 * dx;
						}
						else if (j == Ny - 1) {
							Q_P_v(i, j) += (data.p(i, j - 1) - data.p(i, j)) / 2 * dx;
						}
						else {
							Q_P_v(i, j) += (data.p(i, j - 1) - data.p(i, j + 1)) / 2 * dx;
						}
					}
				}
				arma::mat A_v;
				arma::vec b_v;
				assemble(A_v, b_v, A_W_v, A_S_v, A_P_v, A_N_v, A_E_v, Q_P_v);
				b_v = arma::solve(A_v, b_v);
				reshape(v_star, b_v);
				arma::mat u_f_star = arma::zeros(Nx + 1, Ny);
				for (int i = 1; i < Nx; i++) {
					for (int j = 0; j < Ny; j++) {
						if (i == 1) {
							u_f_star(i, j) = (u_star(i - 1, j) + u_star(i, j)) / 2 - dx * dy / 2 * (1 / A_P_u(i - 1, j) + 1 / A_P_u(i, j)) * ((data.p(i, j) - data.p(i - 1, j)) / dx - ((data.p(i, j) - data.p(i - 1, j)) / (2 * dx) + (data.p(i + 1, j) - data.p(i - 1, j)) / (2 * dx)) / 2);
						}
						else if (i == Nx - 1) {
							u_f_star(i, j) = (u_star(i - 1, j) + u_star(i, j)) / 2 - dx * dy / 2 * (1 / A_P_u(i - 1, j) + 1 / A_P_u(i, j)) * ((data.p(i, j) - data.p(i - 1, j)) / dx - ((data.p(i, j) - data.p(i - 2, j)) / (2 * dx) + (data.p(i, j) - data.p(i - 1, j)) / (2 * dx)) / 2);
						}
						else {
							u_f_star(i, j) = (u_star(i - 1, j) + u_star(i, j)) / 2 - dx * dy / 2 * (1 / A_P_u(i - 1, j) + 1 / A_P_u(i, j)) * ((data.p(i, j) - data.p(i - 1, j)) / dx - ((data.p(i, j) - data.p(i - 2, j)) / (2 * dx) + (data.p(i + 1, j) - data.p(i - 1, j)) / (2 * dx)) / 2);
						}
					}
				}
				arma::mat v_f_star = arma::zeros(Nx, Ny + 1);
				for (int i = 0; i < Nx; i++) {
					for (int j = 1; j < Ny; j++) {
						if (j == 1) {
							v_f_star(i, j) = (v_star(i, j - 1) + v_star(i, j)) / 2 - dx * dy / 2 * (1 / A_P_v(i, j - 1) + 1 / A_P_v(i, j)) * ((data.p(i, j) - data.p(i, j - 1)) / dx - ((data.p(i, j) - data.p(i, j - 1)) / (2 * dx) + (data.p(i, j + 1) - data.p(i, j - 1)) / (2 * dx)) / 2);
						}
						else if (j == Ny - 1) {
							v_f_star(i, j) = (v_star(i, j - 1) + v_star(i, j)) / 2 - dx * dy / 2 * (1 / A_P_v(i, j - 1) + 1 / A_P_v(i, j)) * ((data.p(i, j) - data.p(i, j - 1)) / dx - ((data.p(i, j) - data.p(i, j - 2)) / (2 * dx) + (data.p(i, j) - data.p(i, j - 1)) / (2 * dx)) / 2);
						}
						else {
							v_f_star(i, j) = (v_star(i, j - 1) + v_star(i, j)) / 2 - dx * dy / 2 * (1 / A_P_v(i, j - 1) + 1 / A_P_v(i, j)) * ((data.p(i, j) - data.p(i, j - 1)) / dx - ((data.p(i, j) - data.p(i, j - 2)) / (2 * dx) + (data.p(i, j + 1) - data.p(i, j - 1)) / (2 * dx)) / 2);
						}
					}
				}
				arma::mat p_prime = arma::zeros(Nx, Ny);
				arma::mat A_W_p = arma::zeros(Nx, Ny);
				arma::mat A_S_p = arma::zeros(Nx, Ny);
				arma::mat A_P_p = arma::zeros(Nx, Ny);
				arma::mat A_N_p = arma::zeros(Nx, Ny);
				arma::mat A_E_p = arma::zeros(Nx, Ny);
				arma::mat Q_P_p = arma::zeros(Nx, Ny);
				for (int i = 0; i < Nx; i++) {
					for (int j = 0; j < Ny; j++) {
						if (i == 0 && j == 0) {
							A_P_p(i, j) = 1.0;
						}
						else {
							if (i < Nx - 1) {
								A_P_p(i, j) += (1 / A_P_u(i, j) + 1 / A_P_u(i + 1, j)) / 2 * pow(dy, 2);
								A_E_p(i, j) += -(1 / A_P_u(i, j) + 1 / A_P_u(i + 1, j)) / 2 * pow(dy, 2);
							}
							Q_P_p(i, j) += -u_f_star(i + 1, j) * dy;
							if (i > 0) {
								A_W_p(i, j) += -(1 / A_P_u(i - 1, j) + 1 / A_P_u(i, j)) / 2 * pow(dy, 2);
								A_P_p(i, j) += (1 / A_P_u(i - 1, j) + 1 / A_P_u(i, j)) / 2 * pow(dy, 2);
							}
							Q_P_p(i, j) += u_f_star(i, j) * dy;
							if (j < Ny - 1) {
								A_P_p(i, j) += (1 / A_P_v(i, j) + 1 / A_P_v(i, j + 1)) / 2 * pow(dx, 2);
								A_N_p(i, j) += -(1 / A_P_v(i, j) + 1 / A_P_v(i, j + 1)) / 2 * pow(dx, 2);
							}
							Q_P_p(i, j) += -v_f_star(i, j + 1) * dx;
							if (j > 0) {
								A_S_p(i, j) += -(1 / A_P_v(i, j - 1) + 1 / A_P_v(i, j)) / 2 * pow(dx, 2);
								A_P_p(i, j) += (1 / A_P_v(i, j - 1) + 1 / A_P_v(i, j)) / 2 * pow(dx, 2);
							}
							Q_P_p(i, j) += v_f_star(i, j) * dx;
						}
					}
				}
				arma::mat A_p;
				arma::vec b_p;
				assemble(A_p, b_p, A_W_p, A_S_p, A_P_p, A_N_p, A_E_p, Q_P_p);
				b_p = arma::solve(A_p, b_p);
				reshape(p_prime, b_p);
				arma::mat u_prime = arma::zeros(Nx, Ny);
				for (int i = 0; i < Nx; i++) {
					for (int j = 0; j < Ny; j++) {
						if (i == 0) {
							u_prime(i, j) = (p_prime(i, j) - p_prime(i + 1, j)) / (2 * A_P_u(i, j)) * dy;
						}
						else if (i == Nx - 1) {
							u_prime(i, j) = (p_prime(i - 1, j) - p_prime(i, j)) / (2 * A_P_u(i, j)) * dy;
						}
						else {
							u_prime(i, j) = (p_prime(i - 1, j) - p_prime(i + 1, j)) / (2 * A_P_u(i, j)) * dy;
						}
					}
				}
				arma::mat v_prime = arma::zeros(Nx, Ny);
				for (int i = 0; i < Nx; i++) {
					for (int j = 0; j < Ny; j++) {
						if (j == 0) {
							v_prime(i, j) = (p_prime(i, j) - p_prime(i, j + 1)) / (2 * A_P_v(i, j)) * dx;
						}
						else if (j == Ny - 1) {
							v_prime(i, j) = (p_prime(i, j - 1) - p_prime(i, j)) / (2 * A_P_v(i, j)) * dx;
						}
						else {
							v_prime(i, j) = (p_prime(i, j - 1) - p_prime(i, j + 1)) / (2 * A_P_v(i, j)) * dx;
						}
					}
				}
				arma::mat u_f_prime = arma::zeros(Nx + 1, Ny);
				for (int i = 1; i < Nx; i++) {
					for (int j = 0; j < Ny; j++) {
						u_f_prime(i, j) = (1 / A_P_u(i - 1, j) + 1 / A_P_u(i, j)) * (p_prime(i - 1, j) - p_prime(i, j)) / 2 * dy;
					}
				}
				arma::mat v_f_prime = arma::zeros(Nx, Ny + 1);
				for (int i = 0; i < Nx; i++) {
					for (int j = 1; j < Ny; j++) {
						v_f_prime(i, j) = (1 / A_P_v(i, j - 1) + 1 / A_P_v(i, j)) * (p_prime(i, j - 1) - p_prime(i, j)) / 2 * dx;
					}
				}
				arma::mat u = u_star + u_prime;
				arma::mat v = v_star + v_prime;
				arma::mat u_f = u_f_star + u_f_prime;
				arma::mat v_f = v_f_star + v_f_prime;
				arma::mat p_new = data.p + alpha_p * p_prime;
				arma::mat u_new = alpha_u * u + (1 - alpha_u) * data.u;
				arma::mat v_new = alpha_v * v + (1 - alpha_v) * data.v;
				arma::mat u_f_new = alpha_u * u_f + (1 - alpha_u) * u_f_old;
				arma::mat v_f_new = alpha_v * v_f + (1 - alpha_v) * v_f_old;
				double res_p = arma::abs(data.p - p_new).max();
				double res_u = arma::abs(data.u - u_new).max();
				double res_v = arma::abs(data.v - v_new).max();
				std::cout << "grid=" << grid << " Re=" << Re << " count=" << count << " res_p=" << res_p << " res_u=" << res_u << " res_v=" << res_v << std::endl;
				data.p = p_new;
				data.u = u_new;
				data.v = v_new;
				u_f_old = u_f_new;
				v_f_old = v_f_new;
				if (res_p < tol && res_u < tol && res_v < tol) {
					break;
				}
			}
		}
	}
	void assemble(arma::mat& A, arma::vec& b, arma::mat& A_W, arma::mat& A_S, arma::mat& A_P, arma::mat& A_N, arma::mat& A_E, arma::mat& Q_P) {
		A = arma::zeros(A_P.n_rows * A_P.n_cols, A_P.n_rows * A_P.n_cols);
		for (int i = 0; i < A_P.n_rows; i++) {
			for (int j = 0; j < A_P.n_cols; j++) {
				A(i * A_P.n_cols + j, i * A_P.n_cols + j) = A_P(i, j);
				if (i < A_P.n_rows - 1) {
					A(i * A_P.n_cols + j, (i + 1) * A_P.n_cols + j) = A_E(i, j);
				}
				if (i > 0) {
					A(i * A_P.n_cols + j, (i - 1) * A_P.n_cols + j) = A_W(i, j);
				}
				if (j < A_P.n_cols - 1) {
					A(i * A_P.n_cols + j, i * A_P.n_cols + (j + 1)) = A_N(i, j);
				}
				if (j > 0) {
					A(i * A_P.n_cols + j, i * A_P.n_cols + (j - 1)) = A_S(i, j);
				}
			}
		}
		b = arma::zeros(A_P.n_rows * A_P.n_cols);
		for (int i = 0; i < A_P.n_rows; i++) {
			for (int j = 0; j < A_P.n_cols; j++) {
				b(i * A_P.n_cols + j) = Q_P(i, j);
			}
		}
	}
	void reshape(arma::mat& phi, arma::vec& b) {
		for (int i = 0; i < phi.n_rows; i++) {
			for (int j = 0; j < phi.n_cols; j++) {
				phi(i, j) = b(i * phi.n_cols + j);
			}
		}
	}
public:
	LidDrivenCavityFLow(std::string grid, double Re, int Nx, int Ny, double alpha_p, double alpha_u, double alpha_v, double tol) :grid(grid), Re(Re), Nx(Nx), Ny(Ny), dx(1.0 / Nx), dy(1.0 / Ny), alpha_p(alpha_p), alpha_u(alpha_u), alpha_v(alpha_v), tol(tol) {}
	void solve(Data& data) {
		initialize(data);
		SIMPLE(data);
	}
};
