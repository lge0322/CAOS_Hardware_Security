''' Ring Learning with errors example '''
import numpy as np
import sys
from numpy.polynomial import polynomial as p
from numpy.polynomial import Polynomial

'''Cited from https://asecuritysite.com/encryption/lwe_ring'''

n = 4 # complexity value
q = 2048
xN_1 = [1] + [0] * (n-1) + [1] # X^n + 1
XN_1Poly = Polynomial(xN_1)

def rotate_poly_three(poly):
    first, second, third = poly[1], poly[2], poly[3]
    poly[1], poly[2], poly[3] = third, -second, first
    return poly

def rotate_poly_five(poly):
    first, second, third = poly[1], poly[2], poly[3]
    poly[1], poly[2], poly[3] = -first, second, -third
    return poly

def rotate_poly_three_input_four(poly):
    first, second, third, fourth, fifth, sixth, seventh = poly[1], poly[2], poly[3], poly[4], poly[5], poly[6], poly[7]
    poly[1], poly[2], poly[3], poly[4], poly[5], poly[6], poly[7] = -third, sixth, first, -fourth, seventh, second, -fifth
    return poly

def rotate_poly_five_input_four(poly):
    first, second, third, fourth, fifth, sixth, seventh = poly[1], poly[2], poly[3], poly[4], poly[5], poly[6], poly[7]
    poly[1], poly[2], poly[3], poly[4], poly[5], poly[6], poly[7] = -fifth, -second, seventh, fourth, first, -sixth, -third
    return poly

def rotate_poly_seven_input_four(poly):
    first, second, third, fourth, fifth, sixth, seventh = poly[1], poly[2], poly[3], poly[4], poly[5], poly[6], poly[7]
    poly[1], poly[2], poly[3], poly[4], poly[5], poly[6], poly[7] = -first, second, -third, fourth, -fifth, sixth, -seventh
    return poly

def rotate_poly_nine_input_four(poly):
    first, second, third, fourth, fifth, sixth, seventh = poly[1], poly[2], poly[3], poly[4], poly[5], poly[6], poly[7]
    poly[1], poly[2], poly[3], poly[4], poly[5], poly[6], poly[7] = seventh, -sixth, fifth, -fourth, third, -second, first
    return poly

def rotate_poly_seven(poly):
    first, second, third = poly[1], poly[2], poly[3]
    poly[1], poly[2], poly[3] = -third, -second, -first
    return poly

''' general function for generating a set of polynomial values '''
def generate_polynomial(n, q):
    global xN_1
    l = 0 #Gamma Distribution Location (Mean "center" of dist.)
    poly = np.floor(np.random.normal(l,size=(n)))
    while (len(poly) != n):
        poly = np.floor(np.random.normal(l,size=(n)))
        poly = np.floor(p.polydiv(poly,xN_1)[1]%q)
    # print(type(poly))
    return Polynomial(poly)

''' generate A '''
A = Polynomial(np.floor(np.random.random(size=(n))*q)%q) % xN_1

''' generate B '''
B = Polynomial(np.floor(np.random.random(size=(n))*q)%q) % xN_1

''' generating secret and error polynomials for A '''
s = generate_polynomial(n, q)
e = generate_polynomial(n, q)

pubKey1 = (A*s + e) % q % xN_1
pubKey2 = (B*s + e) % q % xN_1

