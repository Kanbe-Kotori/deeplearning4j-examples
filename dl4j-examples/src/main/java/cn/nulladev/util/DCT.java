package cn.nulladev.util;

public class DCT {

    public static double[] dct(double[] vector) {
        double[] result = new double[vector.length];
        double factor = Math.PI / vector.length;
        for (int i = 0; i < vector.length; i++) {
            double sum = 0;
            for (int j = 0; j < vector.length; j++)
                sum += vector[j] * Math.cos((j + 0.5) * i * factor);
            result[i] = sum * Math.sqrt(2D/vector.length);
        }
        return result;
    }

    public static double[] idct(double[] vector) {
        double[] result = new double[vector.length];
        double factor = Math.PI / vector.length;
        for (int i = 0; i < vector.length; i++) {
            double sum = vector[0] / 2;
            for (int j = 1; j < vector.length; j++)
                sum += vector[j] * Math.cos(j * (i + 0.5) * factor);
            result[i] = sum * Math.sqrt(2D/vector.length);
        }
        return result;
    }
}
