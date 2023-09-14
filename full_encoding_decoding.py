import numpy as np

import math

import random

from numpy.polynomial import Polynomial

from ring_learning_errors import *

'''Cited from https://asecuritysite.com/encryption/lwe_ring'''

''' 
    general idea: 

    (1) computation on encrypted complex vectors --> build an encode and decoder to transform
    our complex vectors into polynomials as encryption, decryption, and other mechanisms work on polynomial rings

    (2) Using the cannonical embedding sigma (decodes a polynomial by evaluating it on the roots of X^N+1)
        able to have an isomorphism between C^N to C[X]/(X^N+1)

    (3) we want out encoder to output polynomials in Z[X]/(X^N+1) --> modify the first vanilla encoder to be able to output polynomials of the right ring
        to exploit the structure of polynomial integer rings

    Any element of sigma(ㅁ) is in a space of dimnsion "N/2" instead of N. 
    Expand the complex vectors of size N/2 when ecoding a vector in CKKS, expand them by copying the other half of conjugate roots

    Method: takes an element of H and projects it to C^(N/2) --> called pi in CKKS

    Note: pi projects, pi^-1 expands
    π^-1(z) ∈ H .

    * coordinate-wise random rounding - to round a real x either to ⌊x⌋  or ⌊x⌋+1 with a probability higher the close x is to either
    --> orthogonal Z-basis
    --> use hermitian product to give real outputs

    * once we have the coordinates Z_i, round them randomly to higher or lower integer to have a polynomial
    that will have integer coordinates in the basis (sigma(1), sigma(X), ... , sigma(X^N-1))

'''

class CKKSEncoder:
    # Basic CKKS encoder to encode complex vectors into polynomials

    def __init__(self, M: int, scale: float):
        self.xi = np.exp(2 * np.pi * 1j / M)
        self.M = M
        self.create_sigma_R_basis()
        self.scale = scale
    
    @staticmethod

    def vandermonde(xi: np.complex128, M: int) -> np.array:
        N = M // 2
        matrix = []
        powers = [pow(5, i, 2*N) for i in range (N//2)]
        for i in range(N//2):
            # print("power: ", powers[i])
            root = xi ** (powers[i])
            row = []
            for j in range(N):
                row.append(root**j)
            matrix.append(row)
        for k in range(N//2):
            # print("minus power: ", -powers[k] % (2*N))
            root = xi ** (-powers[k] % (2*N))
            row = []
            for j in range(N):
                row.append(root**j)
            matrix.append(row)
        return matrix
    
    def sigma_inverse(self, b: np.array) -> Polynomial:
        # encodes the vector b in a polynomial using an M-th root of unity
        A = CKKSEncoder.vandermonde(self.xi, M)
        coeffs = np.linalg.solve(A, b) #solve the system to get an a (coeffs in an array)
        p = Polynomial(coeffs)
        return p
    
    def sigma(self, p: Polynomial) -> np.array:
        # decodes a polynomial by applying it to the M-th roots of unity
        outputs = []
        N = self.M//2
        powers = [pow(5, i, 2*N) for i in range (N//2)]
        for i in range(N//2):
            root = self.xi ** (powers[i])
            output = p(root)
            outputs.append(output)
        for j in range(N//2):
            root = self.xi ** (-powers[j] % (2*N))
            output = p(root)
            outputs.append(output)
        return np.array(outputs)
    
    def pi(self, z: np.array) -> np.array:
        # Projects a vector of H into C^{N/2} 
        N = self.M // 4
        return z[:N]
    
    def pi_inverse(self, z: np.array) -> np.array:
        # Expands a vector of C^{N/2} by expanding it with its complex conjugate
        z_conjugate = z
        z_conjugate = [np.conjugate(x) for x in z_conjugate]
        return np.concatenate([z, z_conjugate])

    def create_sigma_R_basis(self):
        # Crates the basis (sigma(1), sigma(X), ..., sigma(X** N-1))
        self.sigma_R_basis = np.array(self.vandermonde(self.xi, self.M)).T

    def compute_basis_coordinates(self, z):
        # Computes the coordinates of a vector with respect to the orthogonal lattice
        output = np.array([np.real(np.vdot(z, b) / np.vdot(b,b)) for b in self.sigma_R_basis])
        return output

    def coordinate_wise_random_rounding(self, coordinates):
        # rounds coordinates randomly
        r = coordinates - np.floor(coordinates)
        f = np.array([np.random.choice([c, c-1], 1, p=[1-c, c]) for c in r]).reshape(-1)
        rounded_coordinates = coordinates -f
        rounded_coordinates = [int(coeff) for coeff in rounded_coordinates]
        return rounded_coordinates

    def sigma_R_discretization(self, z):
        # projects a vector on the lattice using cooreindate wise random rounding
        coordinates = self.compute_basis_coordinates(z)
        rounded_coordinates = self.coordinate_wise_random_rounding(coordinates)
        y = np.matmul(self.sigma_R_basis.T, rounded_coordinates)
        return y

    def encode(self, z: np.array) -> Polynomial:
        # Encodes a vector by expanding it first to H,
        # scale it, project it on the lattice of sigma(R), and performs
        # sigma inverse.
        pi_z = self.pi_inverse(z)
        # print("pi_z: ", pi_z)
        scaled_pi_z = self.scale * pi_z
        rounded_scale_pi_zi = self.sigma_R_discretization(scaled_pi_z)
        p = self.sigma_inverse(rounded_scale_pi_zi)
        # We round it afterwards due to numerical imprecision
        coef = np.round(np.real(p.coef)).astype(int)
        p = Polynomial(coef)
        return p

    def decode(self, p: Polynomial) -> np.array:
        # Decodes a polynomial by removing the scale, 
        # evaluating on the roots, and project it on C^(N/2)
        rescaled_p = p / self.scale
        z = self.sigma(rescaled_p)
        pi_z = self.pi(z)
        return pi_z

################################################################################

print("####################################################################\n")
print("Example 1: vector of *2* addition WITH CHINESE REMAINDER THEOREM\n")
print("####################################################################\n")

M = 8
N = M //2
scale = 2**30 # 30 int precision
q1 = 1073741839
q2 = 1073741843
q3 = 1073741857
q4 = 1073741891
qi = [q1, q2, q3, q4]
Q_init = 1
# Q_init = q1 * q2 * q3 * q4
for i in range(len(qi)):
    Q_init = Q_init * qi[i]
Q_final = Q_init // qi[len(qi)-1]
xN_1Poly = Polynomial(xN_1)
encoder = CKKSEncoder(M, scale)

# message examples
msgX = np.array([3 + 7j, 1 + 4j])
msgY = np.array([3 + 1j, 2 + 7j])

# encoding messages
encodedX = encoder.encode(msgX)
encodedY = encoder.encode(msgY)

# encrypting message
encryptedX, encryptedX_1 = (encodedX + pubKey1), A
encryptedY, encryptedY_1 = (encodedY + pubKey1), A

# transforming from polynomial to array
def polyToArray(p):
    arr = []
    for elem in p:
        arr.append(elem)
    return np.array(arr)

encryptedX_arr = polyToArray(encryptedX)
encryptedY_arr = polyToArray(encryptedY)
encryptedX_1_arr = polyToArray(encryptedX)
encryptedY_1_arr = polyToArray(encryptedY)
arrLen = len(encryptedX_arr)

# chinese remainder theorem
x = [[0 for _ in range(arrLen)] for _ in range(len(qi))]
y = [[0 for _ in range(arrLen)] for _ in range(len(qi))]
z = [[0 for _ in range(arrLen)] for _ in range(len(qi))]
x_1 = [[0 for _ in range(arrLen)] for _ in range(len(qi))]
y_1 = [[0 for _ in range(arrLen)] for _ in range(len(qi))]
z_1 = [[0 for _ in range(arrLen)] for _ in range(len(qi))]

xN = [[0 for _ in range(arrLen)] for _ in range(len(qi))]
yN = [[0 for _ in range(arrLen)] for _ in range(len(qi))]
zN = [[0 for _ in range(arrLen)] for _ in range(len(qi))]
xN_1 = [[0 for _ in range(arrLen)] for _ in range(len(qi))]
yN_1 = [[0 for _ in range(arrLen)] for _ in range(len(qi))]
zN_1 = [[0 for _ in range(arrLen)] for _ in range(len(qi))]

xN = [[0 for _ in range(arrLen)] for _ in range(len(qi))]
yN = [[0 for _ in range(arrLen)] for _ in range(len(qi))]
zN = [[0 for _ in range(arrLen)] for _ in range(len(qi))]
xN_1 = [[0 for _ in range(arrLen)] for _ in range(len(qi))]
yN_1 = [[0 for _ in range(arrLen)] for _ in range(len(qi))]
zN_1 = [[0 for _ in range(arrLen)] for _ in range(len(qi))]

xK = [[0 for _ in range(arrLen)] for _ in range(len(qi))]
yK = [[0 for _ in range(arrLen)] for _ in range(len(qi))]
zK = [[0 for _ in range(arrLen)] for _ in range(len(qi))]
xK_1 = [[0 for _ in range(arrLen)] for _ in range(len(qi))]
yK_1 = [[0 for _ in range(arrLen)] for _ in range(len(qi))]
zK_1 = [[0 for _ in range(arrLen)] for _ in range(len(qi))]

# changing into RNS form
for i in range(len(qi)-1):
    for j in range(arrLen):
        x[i][j] = int(encryptedX_arr[j] % qi[i])
        xN[i][j] = int(encryptedX_arr[j] % qi[len(qi)-1]) # rescaling
        xK[i][j] = ((x[i][j] - xN[i][j]) * pow(qi[len(qi)-1], -1, qi[i]))
        
        print("Xj", j, xK[i][j])

        y[i][j] = int(encryptedY_arr[j] % qi[i])
        yN[i][j] = int(encryptedY_arr[j] % qi[len(qi)-1]) # rescaling
        yK[i][j] = (y[i][j] - yN[i][j]) * pow(qi[len(qi)-1], -1, qi[i])

        print("Yj", j, yK[i][j])
        z[i][j] = (xK[i][j] + yK[i][j]) % qi[i] # addition

        x_1[i][j] = int(encryptedX_1_arr[j] % qi[i])
        xN_1[i][j] = int(encryptedX_1_arr[j] % qi[len(qi)-1]) # rescaling
        xK_1[i][j] = (x_1[i][j] - xN_1[i][j]) * pow(qi[len(qi)-1], -1, qi[i])

        y_1[i][j] = int(encryptedY_1_arr[j] % qi[i])
        yN_1[i][j] = int(encryptedY_1_arr[j] % qi[len(qi)-1]) # rescaling
        yK_1[i][j] = (y_1[i][j] - yN_1[i][j]) * pow(qi[len(qi)-1], -1, qi[i])
        
        z_1[i][j] = (xK_1[i][j] + yK_1[i][j]) % qi[i] # addition

encrypted_add = [0 for _ in range(arrLen)]
encrypted_add_1 = [0 for _ in range(arrLen)]

# converting back to normal form
for j in range(arrLen): 
    for k in range(len(qi)):
        Qi = Q_init // qi[k]
        Mi = pow(Qi, -1, qi[k])
        encrypted_add[j] = (encrypted_add[j] + (z[k][j] * Qi * Mi)) % Q_init
        encrypted_add_1[j] = (encrypted_add_1[j] + (z_1[k][j] * Qi * Mi)) % Q_init

for j in range(arrLen):
    if (encrypted_add[j] > (Q_init // 2)):
        encrypted_add[j] = encrypted_add[j] - Q_init
    if (encrypted_add_1[j] > (Q_init // 2)):
        encrypted_add_1[j] = encrypted_add_1[j] - Q_init

# decrypting message
decrypted_add = Polynomial(encrypted_add)- (Polynomial(encrypted_add_1) * s) % XN_1Poly % Q_init
decoded_add = encoder.decode(decrypted_add)
print("expected-decoded-addition:", msgX + msgY ,"\n")
print("actual-decoded-addition: ", decoded_add, "\n")


# print("####################################################################\n")
# print("Example 2: vector of *2* multiplications WITH CHINESE REMAINDER THEOREM\n")
# print("####################################################################\n")

# M = 8
# N = M //2
# p = 4096 
# q = 1024 
# scale = 2**30 
# q1 = 1073741839
# q2 = 1073741843
# q3 = 1073741857
# q4 = 1073741891
# qi = [q1, q2, q3, q4]
# Q_init = 1
# for i in range(len(qi)):
#     Q_init = Q_init * qi[i]
# xN_1Poly = Polynomial(xN_1)
# encoder = CKKSEncoder(M, scale)

# # message examples
# msgX = np.array([3 + 7j, 1 + 4j])
# msgY = np.array([3 + 1j, 2 + 7j])

# # encoding messages
# encodedX = encoder.encode(msgX)
# encodedY = encoder.encode(msgY)

# # encrypting message
# encryptedX, encryptedX_1 = (encodedX + pubKey1), A
# encryptedY, encryptedY_1 = (encodedY + pubKey1), A

# # arrays for messages
# encryptedX_arr = polyToArray(encryptedX)
# encryptedY_arr = polyToArray(encryptedY)
# encryptedX_1_arr = polyToArray(encryptedX_1)
# encryptedY_1_arr = polyToArray(encryptedY_1)

# # array lengths
# arrLen = len(encryptedX_arr)
# arrLenPoly = len(encryptedX_arr) * 2 - 1 # without mod
 
# # generating secret keys and coeffs
# s = Polynomial(s)
# s_squared = (s*s) % xN_1Poly
# a0 = Polynomial([random.randint(0,p*q) for _ in range(n)])
# e0 = generate_polynomial(n, q)

# # evaluation keys
# evk0 = (-a0 * s + e0 + p * s_squared) % xN_1Poly
# evk1 = a0 

# # chinese remainder theorem
# x = [[0 for _ in range(arrLen)] for _ in range(len(qi))]
# y = [[0 for _ in range(arrLen)] for _ in range(len(qi))]
# x_1 = [[0 for _ in range(arrLen)] for _ in range(len(qi))]
# y_1 = [[0 for _ in range(arrLen)] for _ in range(len(qi))]

# # len of the array = number of terms before modulus
# d0_r = [[0 for _ in range(arrLenPoly)] for _ in range(len(qi))]
# d1_r = [[0 for _ in range(arrLenPoly)] for _ in range(len(qi))]
# d2_r = [[0 for _ in range(arrLenPoly)] for _ in range(len(qi))]

# # polynomial multiplication function
# def polyMult(A, B):
#     m = len(A)
#     n = len(B)
#     res = [0 for _ in range(m + n - 1)]
#     for i in range(m):
#         for j in range(n):
#             res[i+j] += A[i] * B[j]
#     return res

# # changing into RNS form
# for i in range(len(qi)):
#     for j in range(arrLen):
#         x[i][j] = int(encryptedX_arr[j]) % qi[i]
#         y[i][j] = int(encryptedY_arr[j]) % qi[i]
#         x_1[i][j] = int(encryptedX_1_arr[j]) % qi[i]
#         y_1[i][j] = int(encryptedY_1_arr[j]) % qi[i]
#     d0_r[i] = polyMult(x[i], y[i])
#     d1_r[i] = polyMult(x[i], y_1[i]) + polyMult(x_1[i], y[i])
#     d2_r[i] = polyMult(x_1[i], y_1[i])

# # modding values with qis
# for i in range(len(qi)):
#     for j in range(arrLenPoly):
#         d0_r[i][j] = int(d0_r[i][j]) % qi[i]
#         d1_r[i][j] = int(d1_r[i][j]) % qi[i]
#         d2_r[i][j] = int(d2_r[i][j]) % qi[i]

# # arrays of results for normal form conversion
# encrypted_add_d0 = [0 for _ in range(arrLenPoly)]
# encrypted_add_d1 = [0 for _ in range(arrLenPoly)]
# encrypted_add_d2 = [0 for _ in range(arrLenPoly)]

# # converting back to normal form
# for j in range(arrLenPoly): 
#     for k in range(len(qi)):
#         Qi = Q_init // qi[k]
#         Mi = pow(Qi, -1, qi[k])
#         encrypted_add_d0[j] = (encrypted_add_d0[j] + (d0_r[k][j] * Qi * Mi)) % Q_init
#         encrypted_add_d1[j] = (encrypted_add_d1[j] + (d1_r[k][j] * Qi * Mi)) % Q_init
#         encrypted_add_d2[j] = (encrypted_add_d2[j] + (d2_r[k][j] * Qi * Mi)) % Q_init

# # recognizing the negative values
# for j in range(arrLenPoly):
#     if (encrypted_add_d0[j] > (Q_init // 2)):
#         encrypted_add_d0[j] = encrypted_add_d0[j] - Q_init
#     if (encrypted_add_d1[j] > (Q_init // 2)):
#         encrypted_add_d1[j] = encrypted_add_d1[j] - Q_init
#     if (encrypted_add_d2[j] > (Q_init // 2)):
#         encrypted_add_d2[j] = encrypted_add_d2[j] - Q_init

# # outcomes
# d0 = Polynomial(encrypted_add_d0) % xN_1Poly
# d1 = Polynomial(encrypted_add_d1) % xN_1Poly
# d2 = Polynomial(encrypted_add_d2) % xN_1Poly

# # rescaling
# P0 = ((d2 * evk0) // p) % Q_init % XN_1Poly
# P1 = ((d2 * evk1) // p) % Q_init % XN_1Poly

# c_relin0, c_relin1 = (d0+P0) % xN_1Poly, (d1+P1) % xN_1Poly

# # decrypting message
# decrypted_mul = c_relin0 - (c_relin1 * s) % Q_init % XN_1Poly

# #decoding message
# decoded_mul = encoder.decode(decrypted_mul) / scale
# print("expected decoded_mul:", msgX * msgY, "\n")
# print("decoded_mul", decoded_mul, "\n")