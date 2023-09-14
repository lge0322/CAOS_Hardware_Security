import numpy as np

from numpy.polynomial import Polynomial


''' Cited from https://blog.openmined.org/ckks-explained-part-1-simple-encoding-and-decoding/ '''

'''
    From blog post: https://blog.openmined.org/ckks-explained-part-1-simple-encoding-and-decoding/

    CKKS allows us to perform computations on vectors of complex values (+ real values)
         works with polynomials since providing a good tradeoff b.t. security and efficiency 

    High-level overview:
        1. a message m, a vector of values, is encoded into a plaintext polynomial p(X) and then encrypted using a public key
        2. when m is encrypted into c (a couple of polynomials), CKKS does the operations
        3. decrypting c' = f(c) with the secret key will yield p' = f(p)
        4. Once we decode it, m' = f(m)

    General Idea:
        Implement a homomorphic encryption scheme = have homomorphic properties on encoder, decoder, encryptor, and decrypter
            - operations on c will be decrypted and decoded 

    Encoding in CKKS:
        * encode our input z ∈ C^(N/2) into a polynomial m(X) ∈ Z[X]/(X^N+1) 
        * M = 2N by Φ_M(X) = X^N+1 (m-th cyclotomic polynomial)
        * ㅁ = Z[X]/(X^N+1)
        * ξ_M = M-th root of unity: ξ_M = e^(2i*pi/M)

    Vanila Encoding:
        * encode a vector z ∈ C^N into a polynomial m(X) ∈ Z[X]/(X^N+1)
        * canonical embedding: sigma: C[X]/(X^N+1) -> C^N (decodes and encodes our vectors)
        * to decode a polynomial m(X) into a vector z, evaluate the polynomial on certain values, which 
        will be the roots of the cyclotomic polynomial Φ_M(X) = X^N+1, where N  roots are : ξ,ξ3,...,ξ^2N-1.
        --> Hence, to decode a polynomial m(X), sigma(m)=(m(ξ),m(ξ3),...,m(ξ^2N-1))∈ C^N --> any vector will be uniquely encoded into its corresponding polynomial

        * challenging: encoding of a vector into the corresponding polynomial --> computing the inverse of sigma
        --> find a polynomial m(X) = ∑N-1i=0aiXi ∈mC[X]/(X^N+1) s.t. sigma(m)=(m(ξ),m(ξ3),...,m(ξ^2N-1))=(z1,...,zN)

        * view this as a Vandermonde matrix of the ξ^2i-1

'''
# First we set the parameters
M = 8
N = M //2

# We set xi, which will be used in our computations
xi = np.exp(2 * np.pi * 1j / M)

class CKKSEncoder:
    # Basic CKKS encoder to encode complex vectors into polynomials

    def __init__(self, M: int):
        # Initialization of the encoder for M a power of 2

        # xi = a Mth root of unity, used as a basis for our computation
        self.xi = np.exp(2 * np.pi * 1j / M)
        self.M = M
    
    @staticmethod # Q: what does this do?

    def vandermonde(xi: np.complex128, M: int) -> np.array:
        N = M // 2
        matrix = []
        for i in range(N):
            root = xi ** (2 * i + 1)
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

        for i in range(N):
            root = self.xi ** (2 * i + 1) # odd nums
            output = p(root)
            outputs.append(output)
        return np.array(outputs)

encoder = CKKSEncoder(M)

b = np.array([1,2,3,4]) # encode a vector and see how it's encoded using real values
# print("b:", b)
p = encoder.sigma_inverse(b) # encode the vector into complex polynomials
# print("p:", p)

b_reconstructed = encoder.sigma(p) #extract vector we had initially from the polynomial
# print("b_reconstructed:", b_reconstructed)

np.linalg.norm(b_reconstructed - b) # difference between the values of the reconstruction and inital vector

''' another example '''

m1 = np.array([1,2,3,4])
m2 = np.array([1,-2,3,-4])

p1 = encoder.sigma_inverse(m1)
p2 = encoder.sigma_inverse(m2)

p_add = p1+p2

p_add_res = encoder.sigma(p_add) # p1 + p2 decodes correctly to [2,0,6,0]
# print("p_add_res: ", p_add_res)

''' another example '''

poly_modulo = Polynomial([1,0,0,0,1])
poly_modulo

p_mult = p1 * p2 % poly_modulo # need to do a modulo operation using X^N+1
p_mult_res = encoder.sigma(p_mult)
# print("p_mult_res:", p_mult_res)


