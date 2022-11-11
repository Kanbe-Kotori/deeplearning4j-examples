package cn.nulladev.util;

public class Complex {

    public double real;
    public double imag;

    public Complex(double r, double i){
        this.real = r;
        this.imag = i;
    }

    public Complex(){
        this(0,0);
    }

    public Complex add(Complex x){
        Complex c = new Complex();
        c.real = this.real + x.real;
        c.imag = this.imag + x.imag;
        return c;
    }

    public Complex minus(Complex x){
        Complex c = new Complex();
        c.real = this.real - x.real;
        c.imag = this.imag - x.imag;
        return c;
    }

    public Complex times(Complex x){
        Complex c = new Complex();
        c.real = this.real * x.real - this.imag * x.imag;
        c.imag = this.real * x.imag + this.imag * x.real;
        return c;
    }

    public Complex times(double d){
        Complex c = new Complex();
        c.real = this.real * d;
        c.imag = this.imag * d;
        return c;
    }

    public Complex conjugate() {
        return new Complex(this.real, -this.imag);
    }

    public static void print(Complex a){
        System.out.println(a.real+" + " + a.imag + "i");
    }

    public double abs() {
        return Math.hypot(this.real, this.imag);
    }

    public static double[] absArray(Complex[] complex) {
        double[] res = new double[complex.length];
        for (int i = 0; i < complex.length; i++) {
            res[i] = complex[i].abs();
        }
        return res;
    }

    public static Complex[] convolve(Complex[] x, Complex[] y) {
        if (x.length != y.length) {
            throw new RuntimeException("Dimension don't agree");
        }

        int N = x.length;

        Complex[] a = FFT.fft(x);
        Complex[] b = FFT.fft(y);

        Complex[] c = new Complex[N];
        for (int i = 0; i < N; i++) {
            c[i] = a[i].times(b[i]);
        }

        return c;
    }

}