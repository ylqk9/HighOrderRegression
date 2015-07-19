#include "stdafx.h"
#define MKL_Complex8 std::complex<float>
#define MKL_Complex16 std::complex<double>
#include "mkl.h"

int LinearSolverLAPACK(float *matrix, const int &nrow, float *vec) {
	int n = nrow, nrhs = 1, lda = nrow, ldb = nrow, info = 0;
	int *ipiv = new int[n]();
	info = LAPACKE_sgesv(LAPACK_COL_MAJOR, n, nrhs, matrix, lda, ipiv, vec, ldb);
	delete [] ipiv;
	if(info == 0) return 0;
	else return -1;
}

int LinearSolverLAPACK(double *matrix, const int &nrow, double *vec) {
	int n = nrow, nrhs = 1, lda = nrow, ldb = nrow, info = 0;
	int *ipiv = new int[n]();
	info = LAPACKE_dgesv(LAPACK_COL_MAJOR, n, nrhs, matrix, lda, ipiv, vec, ldb);
	delete [] ipiv;
	if(info == 0) return 0;
	else return -1;
}

int LinearSolverLAPACK(complex<float> *matrix, const int &nrow, complex<float> *vec) {
	int n = nrow, nrhs = 1, lda = nrow, ldb = nrow, info = 0;
	int *ipiv = new int[n]();
	info = LAPACKE_cgesv(LAPACK_COL_MAJOR, n, nrhs, matrix, lda, ipiv, vec, ldb);
	delete [] ipiv;
	if(info == 0) return 0;
	else return -1;
}

int LinearSolverLAPACK(complex<double> *matrix, const int &nrow, complex<double> *vec) {
	int n = nrow, nrhs = 1, lda = nrow, ldb = nrow, info = 0;
	int *ipiv = new int[n]();
	info = LAPACKE_zgesv(LAPACK_COL_MAJOR, n, nrhs, matrix, lda, ipiv, vec, ldb);
	delete [] ipiv;
	if(info == 0) return 0;
	else return -1;
}


template <typename T>
class grid
{
public:
	grid(const int &n, const T &deltax) : ngrid(n), dx(deltax) {};
	~grid(){};
	int				ngrid;
	T				dx;
};


template<typename T>
class Regression
{
public:
	Regression(const grid<T> &g);
	~Regression();
	void				LoadFunction(const string &filename);
	void				Solve(const int &order, T* &Theta);
	void				Solve(const int &order);
	void				Predict(T* y1);
	void				Predict();
	void				Variance();		//order vs variance
	T*					y_reg;
private:
	int					maxorder;
	const grid<T>*		Space;
	T*					gridx;
	T*					y;
	T*					x;
	T*					coeff;
	T*					var;
	int					polyorder;
};

template<typename T>
Regression<T>::Regression(const grid<T> &g) : Space(&g) {
	maxorder = 20;
	gridx = new T[Space->ngrid+1]();
	y = new T[Space->ngrid+1]();
	y_reg = new T[Space->ngrid+1]();
	var = new T[maxorder]();
	for(int i = 0; i < Space->ngrid; ++i) {
		gridx[i] = (i+0.5)*Space->dx;
	}
	coeff = nullptr;
}

template<typename T>
Regression<T>::~Regression() {
	delete [] gridx;
	delete [] y;
	delete [] y_reg;
	delete [] var;
	Space = nullptr;
	if(coeff != nullptr) delete [] coeff;
	if(x != nullptr) delete [] x;
}

template<typename T>
void Regression<T>::LoadFunction(const string &filename) {
	string buff;
	ifstream f1(filename.c_str());
	for(int i = 0; i < Space->ngrid; ++i) {
		getline(f1, buff);
		y[i] = stod(buff);
	}
}

template<typename T>
void Regression<T>::Solve(const int &order, T* &Theta) {
	polyorder = order;
	x = new T[(order + 1)*Space->ngrid]();
	//fill x
	for(int i = 0; i < order; ++i) {
		for(int grd = 0; grd < Space->ngrid; ++grd) {
			x[grd + Space->ngrid*i] = pow(gridx[grd], i+1);
		}
	}
	for(int grd = 0; grd < Space->ngrid; ++grd) x[grd + Space->ngrid*order] = 1;
	int oi, oj;
	//compute x^T*x
	T* xTx = new T[(order + 1)*(order + 1)]();
	for(oi = 0; oi <= order; ++oi) {
		for(oj = oi; oj <= order; ++oj) {
			xTx[oi + oj*(order + 1)] = std::inner_product(x + oi*Space->ngrid, x + oi*Space->ngrid+ Space->ngrid, x + oj*Space->ngrid, 0.0);
			if(oi != oj) xTx[oj + oi*(order + 1)] = xTx[oi + oj*(order + 1)];
		}
	}
	//compute x^T*y
	Theta = new T[order + 1]();
	for(oi = 0; oi <= order; ++oi) {
		Theta[oi] = std::inner_product(x + oi*Space->ngrid, x + (oi + 1)*Space->ngrid, y, 0.0);
	}
	//solve coefficients and store in xTy
	LinearSolverLAPACK(xTx, order + 1, Theta);
}

template<typename T>
void Regression<T>::Solve(const int &order) {
	x = new T[(order + 1)*Space->ngrid]();
	this->Solve(order, this->coeff);
}

template<typename T>
void Regression<T>::Predict(T* y1) {
	for(int i = 0; i < Space->ngrid; ++i) y1[i] = 0.0;
	for(int i = 0; i < Space->ngrid; ++i) {
		for(int j = 0; j <= polyorder; ++j) {
			y1[i] += coeff[j]*x[i + j*Space->ngrid];
		}
	}
}

template<typename T>
void Regression<T>::Predict() {
	this->Predict(this->y_reg);
}

template<typename T>
void Regression<T>::Variance() {
	ofstream order_var("order_var.txt");
	ofstream order_reg("order_reg.txt");
	for(int order = 1; order <= maxorder; ++order) {
		this->Solve(order, coeff);
		for(int i = 0; i <= order; ++i) cout << coeff[i] << " ";
		cout << endl;
		cout << endl;
		//compute y by linear regression
		this->Predict(y_reg);
		for(int i = 0; i < Space->ngrid; ++i) order_reg << y_reg[i] << endl;
		order_reg << endl;
		//compute the variance.
		T* diff = new T[Space->ngrid]();
		for(int i = 0; i < Space->ngrid; ++i) diff[i] = y[i] - y_reg[i];
		var[order - 1] += std::inner_product(diff, diff + Space->ngrid, diff, 0.0);
		order_var << order << " " << var[order - 1] << endl;
		delete [] coeff;
		coeff = nullptr;
		delete [] x;
		x = nullptr;
	}
}
