#include<opencv2/opencv.hpp>

#include <fstream>
#include<iostream>
#include <math.h>
#include <stdio.h>
#include <time.h>

using namespace std;
using namespace cv;


/*****************************************************************************/
/*************************Funciones auxiliares********************************/
/*****************************************************************************/
void pintaMI(vector<Mat> vim){
	
	vector<Mat> aux = vim;


	/* Obtener el tamaño total de la nueva ventana*/
	int totalRows=0, totalCols=0;
	while(!aux.empty()){
		totalCols += aux.back().cols;
		if(aux.back().rows > totalRows)
			totalRows = aux.back().rows;
		aux.pop_back();
	}
	/*********************************************/

	/* Crear la matriz con el nuevo tamaño*/
	Mat big = cvCreateMat(totalRows, totalCols, vim.back().type());

	/* Copiar las imagenes del vector dentro de la pantalla*/
	const int cy =0;
	int cx =0;
	/*Simplemente se hace roi de la imagen grande y se introduce en ella*/
	while(!vim.empty()){
		vim.back().copyTo(big(Rect(cx, cy,vim.back().cols,vim.back().rows)));
		cx += vim.back().cols;
		vim.pop_back();
	}
	/******************************************************/

	/*Mostrar la nueva imagen */
	namedWindow("ventana",1);
	imshow("ventana",big);
	waitKey();
	destroyWindow("ventana");

}

/* funcion para leer imagen:
	flagColor < 0 -> Canal alfa
	flagColor = 0 -> Escala de grises
	flagColor > 0 -> 3 canales
*/
Mat leeimagen(char *filename, int flagColor=1){
	return imread(filename, flagColor);
}


/* Muestra una imagen por pantalla*/
void pintaI(Mat im){
	namedWindow("ventana",1);
	imshow("ventana",im);
	waitKey();
	destroyWindow("ventana");
}

/* Dibuja una seam (linea) en la imagen */
void dibujarLinea(Mat &im, vector<Point> vpoint){

	for(int i=0; i<vpoint.size(); i++){
		im.at<Vec3b>(vpoint[i].x, vpoint[i].y)[0] = 0;
		im.at<Vec3b>(vpoint[i].x, vpoint[i].y)[1] = 0;
		im.at<Vec3b>(vpoint[i].x, vpoint[i].y)[2] = 255;
	
	}
	
}

/*funciones para elegir el minimo de dos o tres variables*/
int minimo2(int n1, int n2){

	if(n2 > n1)
		return n1;
	else
		return n2;
}

int minimo3(int n1, int n2, int n3){
	int min = n3;
	
	if(min > n2)
		min = n2;
	
	if(min > n1)
		min = n1;

	return min;
}

/*Media de la energia de todos los pixeles*/

double mediaGradiente(Mat im){
	double media=0;
	int  *row;

	for(int i=0; i<im.rows; i++){
		row = im.ptr<int>(i);
		for(int j=0; j<im.cols; j++){
			media += row[j];
		}
	}

	return media/(im.cols*im.rows);
}


/*****************************************************************************/
/*************************OBTENER GRADIENTE IMAGEN ***************************/
/*****************************************************************************/
//Dada una imagen se devuelve el gradiente de dicha imagen.
Mat gradienteSobel(Mat im){
	Mat src(im.clone()), src_gray;
	Mat grad;
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;

	GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );

	/// Se convierte la imagen a escala de grises, ya que es mas rapido de utilizar.
	cvtColor( src, src_gray, CV_RGB2GRAY );

	/// Gradiente X mediante sobel y se obtiene el valor absoluto de cada pixel.
	Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
	convertScaleAbs( grad_x, abs_grad_x );

	/// Gradiente Y mediante sobel y se obtiene el valor absoluto de cada pixel.
	Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
	convertScaleAbs( grad_y, abs_grad_y );

	/// El gradiente final sera la suma de los pesos del gradiente x e y.
	addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

	//Se convierte la matriz resultante a tipo int para poder aplicar valores negativos a la hora
	//de eliminar objetos de la escena.
	grad.convertTo(grad, CV_32S);

	return grad;
}


/*****************************************************************************/
/*******************************REDUCIR LA IMAGEN  ***************************/
/*****************************************************************************/
/*

Asigna el valor de un pixel a la siguiente posicion, uso de punteros.

*/
void asignarPixelFila(Mat &im, int i, int j){
	Point3_<uchar>* p = im.ptr<Point3_<uchar> >(i, j);
	Point3_<uchar>* q = im.ptr<Point3_<uchar> >(i+1, j);

	p->x = q->x; //B
	p->y = q->y; //G
	p->z = q->z; //R

	//Forma sin utilizar punteros.

	//im.at<Vec3b>(i,j)[0] = im.at<Vec3b>(i+1,j)[0];
	//im.at<Vec3b>(i,j)[1] = im.at<Vec3b>(i+1,j)[1];
	//im.at<Vec3b>(i,j)[2] = im.at<Vec3b>(i+1,j)[2];
}

void asignarPixelColumna(Mat &im, int i, int j){
	Point3_<uchar>* p = im.ptr<Point3_<uchar> >(i, j);
	Point3_<uchar>* q = im.ptr<Point3_<uchar> >(i, j+1);

	p->x = q->x; //B
	p->y = q->y; //G
	p->z = q->z; //R
}
/*****************************************************/

/*Funciones para borrar una fila o columna*/
/*
	El uso es sencillo, una vez selecionada la linea que se desa borrar (contenida en puntos)
	se asigna el valor del pixel que se va a quitar el valor del siguiente, asi se desplaza
	la linea entera de pixeles. Para terminar se borra la ultima fila o columna.

	Datos de entrada:
	Matriz im: sera la imagen original.
	vector<Point> puntos: un seam que se desee borrar.

	Datos de salida:
	Matriz im: imagen original modificada, columna o fila borrada.
*/

void borrarFila(Mat &im, vector<Point> puntos){
	for(int i=0; i<puntos.size(); i++){	
		for(int j=puntos[i].x; j<im.rows-1; j++){
			asignarPixelFila(im,j,puntos[i].y);
		}
	}
	Mat dst = im.rowRange(0, im.rows-1);
	im = dst;
}

void borrarColumna(Mat &im, vector<Point> puntos){
	for(int i=0; i<puntos.size(); i++){	
		for(int j=puntos[i].y; j<im.cols-1; j++){
			asignarPixelColumna(im,puntos[i].x, j);
		}
	}
	Mat dst = im.colRange(0, im.cols-1);
	im = dst;
}
/*****************************************************/
/*****************************************************/
/*Funciones para crear un mapa dinamico y obtener la columna o fila con menor energia*/

/*
	Dada la funcion M(i,j) = e(i,j) + min(M(i-1,j-1), M(i-1,j), M(i-1,j+1))
	Se crea un mapa dinamico usando el gradiente de toda la imagen.
	En la primera fila o columna se encontrara los valores iniciales, en la ultima
	la energia total. Recomponer el camino es sencillo.

	El acceso a la matriz es mediante punteros donde se obtiene una mejora de cuatro veces
	mejor su rendimiento, ya que se obtiene toda una fila de la imagen y luego se 
	desplaza por ella.

	Dato: Para obtener las filas he utilizado el mismo proceso que para las columnas
	pero con la traspuesta del gradiente esto genera un mejor rendimiento en el tiempo de
	ejecucion.

	Datos de entrada:
	Matriz im: sera el gradiente.
	Matriz M: sera el mapa dinamico donde se alojaran los valores:

	Datos de salida:
	Matriz M modificada.
*/



void mapaDinamicoColumnas(Mat im, Mat &M){
	int min;
	int *rowi, *row;

	for(int i=0; i<im.rows; i++){
		//Obtener la fila i-1 de la matriz.
		if(i>0)
			rowi = M.ptr<int>(i-1);
		//Obtener la fila i de la matriz.
		row = M.ptr<int>(i);
		for(int j=0; j<im.cols; j++){
			int e = im.at<int>(i,j);//Energia del gradiente.
			//si la fila anterior es -1 se dara a esa posicion el valor del gradiente sin ningun otro valor añadido.
			if(i-1 >= 0){
				//Dependiendo del valor del minimo se elige una posicion de M.
				if(j-1 < 0){
					min = minimo2(rowi[j+1], rowi[j]);
				}else if(j+1 >= im.cols){
					min = minimo2(rowi[j-1], rowi[j]);
				}else{
					min = minimo3(rowi[j+1], rowi[j-1], rowi[j]);
				}
				row[j] = e + min;
			}else
				row[j] = e; //Este se iguala a la energia solamente porque es la primera fila
		}
	}
}

void mapaDinamicoFilas(Mat im, Mat &M){
	int min;
	int *rowi, *row;

	////Para crear el mapa dinamico de las filas a quitar se hace el mismo proceso que para el mapa de columnas
	////la unica diferencia es que se hace con la transpuesta del gradiente.
	Mat imt = im.t();
	Mat Mt = M.t();

	for(int i=0; i<imt.rows; i++){

		if(i>0)
			rowi = Mt.ptr<int>(i-1);

		row = Mt.ptr<int>(i);

		for(int j=0; j<imt.cols; j++){
			int e = imt.at<int>(i,j);
			if(i-1 >= 0){
				if(j-1 < 0){
					min = minimo2(rowi[j+1], rowi[j]);
				}else if(j+1 >= im.cols){
					min = minimo2(rowi[j-1], rowi[j]);
				}else{
					min = minimo3(rowi[j+1], rowi[j-1], rowi[j]);
				}
				row[j] = e + min;
			}else
				row[j] = e;
		}
	}
	M = Mt.t();
}


/*****************************************************/
/*****************************************************/

/**** Funciones para componer el minimo camino a partir del mapa dinamico ***/
/*
	Recomponer la linea que menos energia requiere. Para ello se elige el minimo valor de la ultima columna
	o fila y se avanza marcha atras siguiendo los minimos valores, obteniendo la linea con menos coste.

	Datos de entrada:
	Matriz M: mapa dinamico.
	
	Datos de salida:
	vector<Point> un seam con coste minimo encontrado.
*/

vector <Point> construirLineaFilas(Mat M){
	vector <Point> puntos(M.cols);
	int min=99999;
	
	//Se localiza el menor valor obtenido del mapa dinamico.
	for(int i=0; i<M.rows; i++){
		if(min > M.at<int>(i,M.cols-1)){
			min =  M.at<int>(i,M.cols-1);
			puntos[0] = Point(i,M.cols-1);
		}
	}

	//A partir de ese valor se crea la linea marcha atras y eligiendo el menor valor.
	for(int x=0; x+1<M.cols; x++){
		int i = puntos[x].x;
		int j = puntos[x].y;
	
			if(i-1 < 0){
				min = minimo2(M.at<int>(i+1,j-1), M.at<int>(i,j-1));
				if(min == M.at<int>(i+1,j-1))
					puntos[x+1] = Point(i+1, j-1);
				else
					puntos[x+1] = Point(i, j-1);
			}else if(i+1 >= M.rows){
				min = minimo2(M.at<int>(i-1,j-1), M.at<int>(i,j-1));
				if(min == M.at<int>(i-1,j-1))
					puntos[x+1] = Point(i-1,j-1);
				else
					puntos[x+1] = Point(i, j-1);
			}else{
				min = minimo3(M.at<int>(i+1,j-1), M.at<int>(i-1,j-1), M.at<int>(i,j-1));
				if(min == M.at<int>(i+1,j-1))
					puntos[x+1] = Point(i+1, j-1);
				else if(min == M.at<int>(i-1,j-1))
					puntos[x+1] = Point(i-1, j-1);
				else
					puntos[x+1] = Point(i, j-1);
			}
		
	}
	//Se devuelve el seam con el menor coste de energia.
	return puntos;
}

//Igual que la anterior pero para las filas.
vector <Point> construirLineaColumnas(Mat M){
	vector <Point> puntos(M.rows);
	int min=99999;
	

	for(int j=0; j<M.cols; j++){
		if(min > M.at<int>(M.rows-1,j)){
			min =  M.at<int>(M.rows-1,j);
			puntos[0] = Point(M.rows-1,j);
		}
	}

	for(int x=0; x+1<M.rows; x++){
		int i = puntos[x].x;
		int j = puntos[x].y;
	
			if(j-1 < 0){
				min = minimo2(M.at<int>(i-1,j+1), M.at<int>(i-1,j));
				if(min == M.at<int>(i-1,j+1))
					puntos[x+1] = Point(i-1, j+1);
				else
					puntos[x+1] = Point(i-1, j);
			}else if(j+1 >= M.cols){
				min = minimo2(M.at<int>(i-1,j-1), M.at<int>(i-1,j));
				if(min == M.at<int>(i-1,j-1))
					puntos[x+1] = Point(i-1,j-1);
				else
					puntos[x+1] = Point(i-1, j);
			}else{
				min = minimo3(M.at<int>(i-1,j+1), M.at<int>(i-1,j-1), M.at<int>(i-1,j));
				if(min == M.at<int>(i-1,j+1))
					puntos[x+1] = Point(i-1, j+1);
				else if(min == M.at<int>(i-1,j-1))
					puntos[x+1] = Point(i-1, j-1);
				else
					puntos[x+1] = Point(i-1, j);
			}
		
	}
	return puntos;
}

/*****************************************************/
/*****************************************************/

/*
	Funcion principal, reducir la imagen alternando entre fila y columna

	Datos de entrada:
	Matriz im1: imagen para modificar.
	const int filas: cantidad de filas que se desean borrar.
	const int columnas: cantidad de columnas que se desean borrar.
	
	Datos de salida:
	Matriz im1: imagen original modificada.
	
*/

void reducirImagenAlternando(Mat &im1, const int filas, const int columnas){
	int f = 0;
	int c = 0;
	bool filOcol = true;
	vector<Point> puntosFil;
	vector<Point> puntosCol;

	for(int i=0; i<filas+columnas; i++){
		//Se crea un mapa dinamico y se obtiene un gradiente, se puede utilizar
		//tanto para columnas como para filas a si que se puede crear fuera.
		Mat M(im1.rows, im1.cols, CV_32S);
		Mat gradiente = gradienteSobel(im1.clone());

		//Valores bool para alternar entre filas y columnas.
		if(columnas<=c)
			filOcol = true;
		else if(filas<=f)
			filOcol = false;

		if(filOcol == true){
			//Algoritmo principal para borrar filas.
			//Se crea un mapa dinamico, se obtiene el seam con menor coste y se borra.
			mapaDinamicoFilas(gradiente, M);
			puntosFil= construirLineaFilas(M);
			borrarFila(im1, puntosFil);

			f++;
			filOcol = false;
		}else if(filOcol == false){
			//Al igual que para las filas pero con columnas.
			mapaDinamicoColumnas(gradiente, M);
			puntosCol = construirLineaColumnas(M);
			borrarColumna(im1, puntosCol);

			c++;
			filOcol = true;
		}
	}
}


/*Funcion principal, reducir la imagen primero eliminar las filas y luego columnas
	Datos de entrada:
	Matriz im1: imagen para modificar.
	const int filas: cantidad de filas que se desean borrar.
	const int columnas: cantidad de columnas que se desean borrar.
	
	Datos de salida:
	Matriz im1: imagen original modificada.
*/

void reducirImagenPrimeroFilas(Mat &im1, const int filas, const int columnas){
	bool filOcol = true;
	vector<Point> puntosFil;
	vector<Point> puntosCol;

	for(int i=0; i<filas; i++){
		Mat M(im1.rows, im1.cols, CV_32S);
		Mat gradiente = gradienteSobel(im1.clone());
		mapaDinamicoFilas(gradiente, M);
		puntosFil= construirLineaFilas(M);
		borrarFila(im1, puntosFil);
	}

	for(int i=0; i<columnas; i++){
		Mat M(im1.rows, im1.cols, CV_32S);
		Mat gradiente = gradienteSobel(im1.clone());
		mapaDinamicoColumnas(gradiente, M);
		puntosCol = construirLineaColumnas(M);
		borrarColumna(im1, puntosCol);
	}
}

/*Funcion principal, reducir la imagen primero eliminar las columnas y despues las filas
	Datos de entrada:
	Matriz im1: imagen para modificar.
	const int filas: cantidad de filas que se desean borrar.
	const int columnas: cantidad de columnas que se desean borrar.
	
	Datos de salida:
	Matriz im1: imagen original modificada.
*/

void reducirImagenPrimeroColumnas(Mat &im1, const int filas, const int columnas){
	bool filOcol = true;
	vector<Point> puntosFil;
	vector<Point> puntosCol;

	for(int i=0; i<columnas; i++){
		Mat M(im1.rows, im1.cols, CV_32S);
		Mat gradiente = gradienteSobel(im1.clone());
		mapaDinamicoColumnas(gradiente, M);
		puntosCol = construirLineaColumnas(M);
		borrarColumna(im1, puntosCol);
	}

	for(int i=0; i<filas; i++){
		Mat M(im1.rows, im1.cols, CV_32S);
		Mat gradiente = gradienteSobel(im1.clone());
		mapaDinamicoFilas(gradiente, M);
		puntosFil= construirLineaFilas(M);
		borrarFila(im1, puntosFil);
	}

}


/*****************************************************************************/
/***************************OPTIMAL ORDER*************************************/
/*****************************************************************************/

//De un mapa dinamico M se obtiene el seam con el menor coste para una columna.
int obtenerEnergiaCol(Mat M){
	int min=99999;
	for(int j=0; j<M.cols; j++){
		if(min > M.at<int>(M.rows-1,j)){
			min =  M.at<int>(M.rows-1,j);
		}
	}
	return min;
}

//De un mapa dinamico M se obtiene el seam con el menor coste para una fila.
int obtenerEnergiaFil(Mat M){
	int min=99999;
	for(int i=0; i<M.rows; i++){
		if(min > M.at<int>(i,M.cols-1)){
			min =  M.at<int>(i,M.cols-1);
		}
	}
	return min;
}


//Obtener la menor energia de una columna, en forma de entero. Ademas borra esa columna de imAux.
int obtenerEnergiaColumna(Mat &imAux){
	Mat M(imAux.rows, imAux.cols, CV_32S);
	Mat gradiente = gradienteSobel(imAux.clone());
	mapaDinamicoColumnas(gradiente, M);
	
	//Dado el mapa dinamico se obtiene cual es el seam con menor coste.
	int min=obtenerEnergiaCol(M);

	//Se borra dicho seam.
	vector<Point> puntosCol = construirLineaColumnas(M);
	borrarColumna(imAux, puntosCol);

	//Se devuelve el coste de borrado del seam.
	return min;
}

//Obtener la menor energia de una fila, en forma de entero. Aparte borra esa fila de imAux.
//Igual que la anterior pero para filas.
int obtenerEnergiaFila(Mat &imAux){
	Mat M(imAux.rows, imAux.cols, CV_32S);
	Mat gradiente = gradienteSobel(imAux.clone());
	mapaDinamicoFilas(gradiente, M);

	int min=obtenerEnergiaFil(M);

	vector<Point> puntosFil = construirLineaFilas(M);
	borrarFila(imAux, puntosFil);
	return min;

}

//Busca una matriz de rango rxc dentro de ims.
Mat buscarMat(vector<Mat> &ims,int r,int c){
	for(vector<Mat>::iterator it=ims.begin(); it!=ims.end(); it++){
		if((*it).rows == r && (*it).cols == c){
			Mat aux = (*it).clone();
			ims.erase(it);
			return aux ;
		}
	}
	return Mat();
}

/*
	Funcion principal, reducir la imagen buscando el orden optimo.

	Datos de entrada:
	Matriz im1: imagen para modificar.
	const int filas: cantidad de filas que se desean borrar.
	const int columnas: cantidad de columnas que se desean borrar.
	
	Datos de salida:
	Matriz im1: imagen original modificada.

	Este algoritmo sigue la funcion:
	T(r,c) = min(T(r-1,c)+E(sx(I n-r-1xm-c)), T(r,c-1)+E(sy(I n-rxm-c-1)))
	Implementada mediante programacion dinamica.
	
	Cada I es una matriz de dimensiones nxm, para no tener que estar calculandola continuamente se guardaran en un vector
	y cuando se desee reducir se buscara la matriz adecuada.
*/

void reducirImagenesOptimalOrder(Mat &im, const int filas, const int columnas){
	Mat T(filas+1, columnas+1, CV_32S);
	//Primer elemento, no se borra columna ni fila por tanto la energia es 0.
	T.at<int>(0,0) = 0;

	vector<Mat> ims;
	ims.push_back(im.clone());
	ims.push_back(im.clone());

	for(int i=0; i<T.rows; i++){
		for(int j=0; j<T.cols; j++){
			if(!(i==0 && j==0)){
				if(j-1 < 0){
					//Se busca la matriz Inxm que se busca y se utilizara para borrar la columna o fila.
					Mat imCol = buscarMat(ims, im.rows-i+1, im.cols-j);
					T.at<int>(i,j) = T.at<int>(i-1,j) + obtenerEnergiaFila(imCol);
					//Se introducen dos copias del resultado de borrar una fila para utilizarla en otra ocasion, 
					//el porque de dos copias se debe a que cada matriz se utiliza dos veces.
					ims.push_back(imCol.clone());
					ims.push_back(imCol.clone());
				}else if(i-1 < 0){
					//Se busca la matriz Inxm que se busca y se utilizara para borrar la columna o fila.
					Mat imFil= buscarMat(ims, im.rows-i, im.cols-j+1);	
					T.at<int>(i,j) = T.at<int>(i,j-1) + obtenerEnergiaColumna(imFil);
					ims.push_back(imFil.clone());
					ims.push_back(imFil.clone());
				}else{
					Mat imCol = buscarMat(ims,im.rows-i+1, im.cols-j);
					Mat imFil= buscarMat(ims, im.rows-i, im.cols-j+1);
					int min1 = T.at<int>(i-1,j) + obtenerEnergiaFila(imCol);
					int min2 = T.at<int>(i,j-1) + obtenerEnergiaColumna(imFil);

					if(min1 >= min2){
						T.at<int>(i,j) = min2;
						ims.push_back(imFil);
						ims.push_back(imFil);
					}else{
						T.at<int>(i,j) = min1;
						ims.push_back(imCol);
						ims.push_back(imCol);
					}

				}
			}
		}
	}
	//Para finalizar se busca la ultima matriz modificada que sera la optima.
	im = buscarMat(ims,im.rows-filas,im.cols-columnas);
}
/*
	Funcion principal, reducir la imagen buscando el orden optimo local.

	Datos de entrada:
	Matriz im1: imagen para modificar.
	const int filas: cantidad de filas que se desean borrar.
	const int columnas: cantidad de columnas que se desean borrar.
	
	Datos de salida:
	Matriz im1: imagen original modificada.

	Comprueba cada seam fila o columna antes de borrarlo el coste de energia, el menor sera borrado.
	
*/

void reducirImagenOptimalOrderLocal(Mat &im1, const int filas, const int columnas){
	int f = 0;
	int c = 0;
	bool fil = true;
	bool col = true;
	int Ef=0, Ec=0;
	vector<Point> puntosFil;
	vector<Point> puntosCol;

	for(int i=0; i<filas+columnas; i++){
		Mat Mf(im1.rows, im1.cols, CV_32S);
		Mat Mc(im1.rows, im1.cols, CV_32S);
		Mat gradiente = gradienteSobel(im1.clone());

		//Si se excede el numero de filas o columnas que se desea borrar
		//el tipo bool tornara a false para no volver a borrar la fila o columna.
		if(f >= filas)
			fil = false;
		if(c >= columnas)
			fil = false;

		//Si no se excedio el maximo permitido de borrado se obtiene el mapa dinamico de
		//las filas y de las columnas, se obtiene la energia.
		if(fil==true){
			mapaDinamicoFilas(gradiente, Mf);
			Ef = obtenerEnergiaFil(Mf);
		}
		if(col==true){
			mapaDinamicoColumnas(gradiente, Mc);
			Ec = obtenerEnergiaCol(Mc);
		}


		//Si la energia de la fila es mayor se borra la columna y viceversa.
		if((Ef < Ec && fil == true) || col == false){	
			puntosFil= construirLineaFilas(Mf);
			borrarFila(im1, puntosFil);
			f++;
		}else if((Ef >= Ec && col == true) || fil == false){	
			puntosCol = construirLineaColumnas(Mc);
			borrarColumna(im1, puntosCol);
			c++;
		}
	}
}

/**********************************************************************************/
/***************************AUMENTAR LA IMAGEN*************************************/
/**********************************************************************************/

/*Obtiene los colores de un pixel en un vector
	Datos de entrada:
	Matriz im: imagen.
	int i: fila.
	int j: columna.
	
	Datos de salida:
	Vec3b: valores del pixel,
*/
Vec3b getColores(Mat im, int i, int j){
	Vec3b RGB(3);
	//RGB[0] = im.at<Vec3b>(i,j)[0];
	//RGB[1] = im.at<Vec3b>(i,j)[1];
	//RGB[2] = im.at<Vec3b>(i,j)[2];
	Point3_<uchar>* p = im.ptr<Point3_<uchar> >(i, j);

	RGB[0] = p->x; //B
	RGB[1] = p->y; //G
	RGB[2] = p->z; //R
	return RGB;

}
/*	Duplica una fila dentro de la matriz.
	Utilizando un seam abre una nueva posicion para cada punto y lo rellena con la media de
	los valores de los pixeles superior e inferior.

	Datos de entrada:
	Matriz im1: imagen para modificar.
	vector<Point> un seam con coste minimo encontrado.
	
	Datos de salida:
	Matriz im1: imagen original modificada.
*/


void duplicarFila(Mat &im, vector<Point> puntos){
	Mat dst(im.rows+1, im.cols, im.type());
	im.copyTo(dst(Rect(Point(0, 1), im.size())));
	int c0, c1, c2;
	Vec3b RGB1, RGB2;
	

	for(int i=0; i<puntos.size(); i++){
		//Abre una posicion extra para cada pixel del seam.
		//Se asigna a la actual posicion i la siguiente, asi sucesivamente hasta terminar.
		for(int j=0; j<puntos[i].x+1; j++){
			asignarPixelFila(dst,j,puntos[i].y);
		}

		//Estructura de control para que no salga del rango.
		if((dst.rows > puntos[i].x + 1) && (0 <= puntos[i].x - 1)){
			//Se obtienen los colores del pixel superior e inferior.
			RGB1 = getColores(dst, puntos[i].x+1,puntos[i].y);
			RGB2 = getColores(dst, puntos[i].x-1, puntos[i].y);
		
			//Se suman y hacen la media.
			c0 = (RGB1[0] + RGB2[0])/2;
			c1 = (RGB1[1] + RGB2[1])/2;
			c2 = (RGB1[2] + RGB2[2])/2;

			//Se aplican al nuevo pixel.
			dst.at<Vec3b>(puntos[i].x,puntos[i].y)[0] = c0;
			dst.at<Vec3b>(puntos[i].x,puntos[i].y)[1] = c1;
			dst.at<Vec3b>(puntos[i].x,puntos[i].y)[2] = c2;
		}
	}
	im = dst;
}

/*	Duplica una columna dentro de la matriz.
	Utilizando un seam abre una nueva posicion para cada punto y lo rellena con la media de
	los valores de los pixeles derecho e izquierdo.

	Datos de entrada:
	Matriz im1: imagen para modificar.
	vector<Point> un seam con coste minimo encontrado.
	
	Datos de salida:
	Matriz im1: imagen original modificada.

	El proceso es igual que el anterior.
*/

void duplicarColumna(Mat &im, vector<Point> puntos){
	Mat dst(im.rows, im.cols+1, im.type());
	im.copyTo(dst(Rect(Point(1, 0), im.size())));
	int c0, c1, c2;
	Vec3b RGB1, RGB2;

	for(int i=0; i<puntos.size(); i++){
		for(int j=0; j<puntos[i].y+1; j++){
			asignarPixelColumna(dst, puntos[i].x, j);
		}

		if((dst.cols > puntos[i].y + 1) && (0 <= puntos[i].y - 1)){
			RGB1 = getColores(dst, puntos[i].x, puntos[i].y+1);
			RGB2 = getColores(dst, puntos[i].x, puntos[i].y-1);
		
			c0 = (RGB1[0] + RGB2[0])/2;
			c1 = (RGB1[1] + RGB2[1])/2;
			c2 = (RGB1[2] + RGB2[2])/2;

			dst.at<Vec3b>(puntos[i].x,puntos[i].y)[0] = c0;
			dst.at<Vec3b>(puntos[i].x,puntos[i].y)[1] = c1;
			dst.at<Vec3b>(puntos[i].x,puntos[i].y)[2] = c2;
		}
	}
	im = dst;
}

/*Funciones auxiliare para desplazar una columna o fila dada una constante c*/
void aumentarLineaColumnas(vector<Point> &puntos, const int c){
	for(int i=0; i<puntos.size(); i++){
		puntos[i].y+=c;
	}
}

void aumentarLineaFilas(vector<Point> &puntos, const int c){
	for(int i=0; i<puntos.size(); i++){
		puntos[i].x+=c;
	}
}

/***********************************************/

/*Con estas dos funciones se desplaza la columna o fila x veces has obtener su posicion absoluta en la imagen original*/
void aumentarTodasFilas(vector<vector<Point>> &seams, vector<Point> linea){
	vector<vector<Point>> seamsRemoved;
	while(!seams.empty()){
		vector<Point> aux = seams.back();
		//Depende de si la fila esta antes o despues de la ultima fila borrada significara que estaba
		//una posicion por delante o por detras, si estaba delante con sumarle 1 a todos los puntos del
		//seam bastara para alcanzar la posicion absoluta.
		if(linea[linea.size()-1].x <= aux[aux.size()-1].x)
			aumentarLineaFilas(aux, 1);

		seamsRemoved.push_back(aux);
		seams.pop_back();
	}
	seams = seamsRemoved;
}

void aumentarTodasColumnas(vector<vector<Point>> &seams, vector<Point> linea){
	vector<vector<Point>> seamsRemoved;
	while(!seams.empty()){
		vector<Point> aux = seams.back();
		if(linea[linea.size()-1].y <= aux[aux.size()-1].y)
			aumentarLineaColumnas(aux, 1);

		seamsRemoved.push_back(aux);
		seams.pop_back();
	}
	seams = seamsRemoved;
}
/*******************************************************************************************************/

/*Funcion principal para aumentar la imagen dada*/
void aumentarImagen(Mat &im1, const int filas, const int columnas){

	//Segun la teoria de seam carving el aumentar la imagen un 50% mas de su tamaño original puede causar resultos raros.
	if((filas > im1.rows/2) || (columnas > im1.cols/2))
		cout << "AVISO: aumentar la imagen mas de un 50% de la dimension original podria causar resultados inesperados" << endl;

	vector<Point> lineaAux;
	/*************Duplicar columnas****************/
	vector<vector<Point>> todosSeamsColumnas;
	Mat imAux = im1.clone();
	vector<Point> puntosCol;

	//Primero se obtienen que columnas se borrarian y se guardan en un vector.
	for(int i=0; i<columnas; i++){
		Mat M(imAux.rows, imAux.cols, CV_32S);
		Mat gradiente = gradienteSobel(imAux.clone());
		mapaDinamicoColumnas(gradiente, M);
		puntosCol = construirLineaColumnas(M);

		todosSeamsColumnas.push_back(puntosCol);
		borrarColumna(imAux, puntosCol);
	}


	//Ahora se recomponen las columnas, es decir la posicion relativa del seam quitado no es la
	//misma que la posicion absoluta de la image, para ello se recompone las columnas o filas
	//utilizando la seam quitada anteriormente, si esta una posicion por delante su posicion absoluta
	//y relativa sera la misma, pero si esta una posicion por detras habra que desplazar la columna o fila
	//una posicion hacia delante.


	vector<vector<Point>> seamsQuitadasC;
	seamsQuitadasC.push_back(todosSeamsColumnas.back());
	todosSeamsColumnas.pop_back();

	while(!todosSeamsColumnas.empty()){
		lineaAux = todosSeamsColumnas.back();
		aumentarTodasColumnas(seamsQuitadasC, lineaAux);
		seamsQuitadasC.push_back(lineaAux);
		todosSeamsColumnas.pop_back();
	}

	//Una vez se sabe la posicion absoluta de todas las seams se duplican en la imagen original.

	while(!seamsQuitadasC.empty()){
		duplicarColumna(im1, seamsQuitadasC.back());
		lineaAux = seamsQuitadasC.back();
		seamsQuitadasC.pop_back();
		aumentarTodasColumnas(seamsQuitadasC, lineaAux);
	}

	/*************Duplicar filas******************/

	//Igual que el proceso anterior solo que esta vez se obtienen las filas a borrar.
	imAux = im1.clone();
	vector<Point> puntosFil;
	vector<vector<Point>> todosSeamsFilas;

	for(int i=0; i<filas; i++){
		Mat M(imAux.rows, imAux.cols, CV_32S);
		Mat gradiente = gradienteSobel(imAux.clone());
		mapaDinamicoFilas(gradiente, M);
		puntosFil = construirLineaFilas(M);
		todosSeamsFilas.push_back(puntosFil);
		borrarFila(imAux, puntosFil);
	}
	//Se recomponen las filas, de posicion relativa a posicion absoluta.
	vector<vector<Point>> seamsQuitadasF;
	seamsQuitadasF.push_back(todosSeamsFilas.back());
	todosSeamsFilas.pop_back();
	while(!todosSeamsFilas.empty()){
		vector<Point> lineaAux = todosSeamsFilas.back();
		aumentarTodasFilas(seamsQuitadasF, lineaAux);
		seamsQuitadasF.push_back(lineaAux);
		todosSeamsFilas.pop_back();
	}
	//Se duplican las filas.
	while(!seamsQuitadasF.empty()){
		duplicarFila(im1, seamsQuitadasF.back());
		lineaAux = seamsQuitadasF.back();
		seamsQuitadasF.pop_back();
		aumentarTodasFilas(seamsQuitadasF, lineaAux);
	}

}



/*****************************************************************************/
/***************************OBJECT REMOVAL ***********************************/
/*****************************************************************************/


//Comprueba que el pixel esta marcado, en mi caso los pixeles que sean totalmente rojos (0,0,255).
//Datos de entrada una imagen im y un punto p a analizar
//Return true si es el pixel a eliminar.
//Return false si no es el pixel.
bool compararPixeles(Mat im, Point p){

	if( im.at<Vec3b>(p.x, p.y)[0] == 0 && 
		im.at<Vec3b>(p.x, p.y)[1] == 0 &&
	    im.at<Vec3b>(p.x, p.y)[2] == 255)
		return true;
	else
		return false;

}

//Busca los puntos que estan marcados dentro de la imagen, si los encuentra los introduce en puntosEliminar.
vector<Point> obtenerPixelesEliminar(Mat im){
	vector <Point> puntosEliminar;

	for(int i=0; i<im.rows; i++){
		for(int j=0; j<im.cols; j++){
			if(compararPixeles(im, Point(i,j)))
				puntosEliminar.push_back(Point(i,j));
		}
	}

	return puntosEliminar;
}

/* Funcion principal
	Llamar a esta funcion supone dada una imagen im, buscara los puntos marcados e intentara elimarlos.
	Esta funcion solamente borrara columnas.
	Entrada:
	Matriz im: imagen con los puntos marcados.
	Salida:
	Matriz im modificada.
*/

void eliminarPuntosColumnas(Mat &im){
	vector<Point> puntosCol, puntosFil;
	vector<Point> puntosEliminar = obtenerPixelesEliminar(im);
	bool filOcol = true;
	//Comprueba que no hay ningun punto para eliminar dentro del vector
	while(!puntosEliminar.empty()){
		Mat M(im.rows, im.cols, CV_32S);
		Mat gradiente = gradienteSobel(im.clone());
		//Marca en el gradiente los puntos seleccionados como negativos.
		for(vector<Point>::iterator it=puntosEliminar.begin(); it!=puntosEliminar.end(); it++){
			gradiente.at<int>((*it).x, (*it).y) = -2000;
		}
		//Ahora se borran las columnas, este proceso buscara preferentemente los valores negativos.
		mapaDinamicoColumnas(gradiente, M);
		puntosCol = construirLineaColumnas(M);
		borrarColumna(im, puntosCol);

		//Refresca el vector con los numeros para eliminar
		puntosEliminar = obtenerPixelesEliminar(im);
		
	}
}

/* Funcion principal
	Llamar a esta funcion supone dada una imagen im, buscara los puntos marcados e intentara elimarlos.
	Esta funcion alterna entre columnas y filas para borrar los puntos.
	Entrada:
	Matriz im: imagen con los puntos marcados.
	Salida:
	Matriz im modificada.
*/

void eliminarPuntosAlternando(Mat &im){
	
	vector<Point> puntosCol, puntosFil;
	vector<Point> puntosEliminar = obtenerPixelesEliminar(im);
	bool filOcol = true;
	//Comprueba que no hay ningun punto para eliminar dentro del vector
	while(!puntosEliminar.empty()){	
		Mat M(im.rows, im.cols, CV_32S);
		Mat gradiente = gradienteSobel(im.clone());
		//Marca en el gradiente los puntos seleccionados como negativos.
		for(vector<Point>::iterator it=puntosEliminar.begin(); it!=puntosEliminar.end(); it++){
			gradiente.at<int>((*it).x, (*it).y) = -2000;
		}
		//Borra primero una fila y luego columna sucesivamente hasta terminar.
		if(filOcol == true){
			mapaDinamicoFilas(gradiente, M);
			puntosFil= construirLineaFilas(M);
			borrarFila(im, puntosFil);
			filOcol = false;
		}else if(filOcol == false){
			mapaDinamicoColumnas(gradiente, M);
			puntosCol = construirLineaColumnas(M);
			borrarColumna(im, puntosCol);
			filOcol = true;
		}
		//Refresca el vector con los numeros para eliminar
		puntosEliminar = obtenerPixelesEliminar(im);
		
	}

}

int main (){
	Mat im1 = leeimagen("imagenes/roca.jpg",1);
	
	Mat im3 = leeimagen("imagenes/birdRemoved2.bmp",1);
	Mat im2 = leeimagen("imagenes/bird.jpg",1);


	//Mat im1 = leeimagen("imagenes/rocaRemoved.bmp",1);
	//Mat im1 = leeimagen("imagenes/peq.jpg",1);
	//Mat im1 = leeimagen("imagenes/marino.jpg",1);
	//Mat im1 = leeimagen("imagenes/birdRemoved1.bmp",1);
	//Mat im1 = leeimagen("imagenes/redRock.jpg",1);
	//Mat im1 = leeimagen("imagenes/zeus.jpg",1);
	//Mat im1 = leeimagen("imagenes/casa.jpg",1);

	if(im1.empty() || im2.empty() || im3.empty()){
		cout << "No se encuentra alguna imagen!" << endl;
		system("Pause");
		return -1;
	}

	const int columnas = 50;
	const int filas = 50;

	if(im1.rows <= filas || im1.cols <= columnas){
		cout << "No se pueden eliminar mas filas o columnas de las que existen!" << endl;
		system("Pause");
		return -1;

	}
	Mat imAux1;
	vector<Mat> ims1;

	/*Reducir la imagen 50 filas y 50 columnas alternando entre fila y columna*/
	cout << "Ahora se procedera a reducir la imagen en " <<filas<< " filas y "<<columnas<<" columnas" << endl;
	cout << "El metodo usado sera alternando entre filas y columnas" << endl;
	cout << "Aviso: el proceso podria tardar entre 10 y 20 segundos" << endl;
	cout << "Dimension original: " << im1.rows << " " << im1.cols << endl;
	clock_t start=clock();
	imAux1 = im1.clone();
	reducirImagenAlternando(imAux1, filas, columnas);
	ims1.push_back(im1);
	ims1.push_back(imAux1);
	cout << "Dimension final: " << imAux1.rows << " " << imAux1.cols << endl;
	cout << "Tiempo total: " << ((double)clock() - start) / CLOCKS_PER_SEC << endl;
	pintaMI(ims1);
	ims1.clear();

	/*Aumentar la imagen 50 filas y 50 columnas*/
	cout << endl << endl;
	cout << "Ahora se procedera a aumentar la imagen en " <<filas<< " filas y "<<columnas<<" columnas" << endl;
	cout << "Aviso: el proceso podria tardar entre 10 y 30 segundos" << endl;
	cout << "Dimension original: " << im1.rows << " " << im1.cols << endl;
	start=clock();
	imAux1 = im1.clone();
	aumentarImagen(imAux1, filas, columnas);
	ims1.push_back(im1);
	ims1.push_back(imAux1);
	cout << "Dimension final: " << imAux1.rows << " " << imAux1.cols << endl;
	cout << "Tiempo total: " << ((double)clock() - start) / CLOCKS_PER_SEC << endl;
	pintaMI(ims1);
	ims1.clear();


	/*Borrar el objeto de la escena*/
	cout << endl << endl;
	cout << "Ahora se procedera a eliminar el objeto de la escena" << endl;
	cout << "Aviso: el proceso podria llegar a tardar 60 segundos" << endl;
	cout << "El metodo usado sera borrar columnas y filas" << endl;
	cout << "Dimension original: " << im1.rows << " " << im1.cols << endl;
	start=clock();
	imAux1 = im3.clone();
	eliminarPuntosAlternando(imAux1);
	ims1.push_back(im2);
	ims1.push_back(imAux1);
	cout << "Dimension final: " << imAux1.rows << " " << imAux1.cols << endl;
	cout << "Tiempo total: " << ((double)clock() - start) / CLOCKS_PER_SEC << endl;
	pintaMI(ims1);
	ims1.clear();


	return 0;
}

