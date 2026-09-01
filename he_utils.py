import tenseal as ts

def create_context():
    """
    Creates the CKKS encryption context.
    poly_modulus_degree and coeff_mod_bit_sizes control precision vs speed.
    global_scale controls decimal precision retained through computation.
    """
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.generate_galois_keys()
    context.global_scale = 2**40
    return context

def encrypt_vector(context, plain_vector):
    """Encrypts a list/array of numbers into a single CKKS ciphertext vector."""
    return ts.ckks_vector(context, plain_vector)

def decrypt_vector(encrypted_vector):
    """Decrypts a CKKS ciphertext vector back into plaintext numbers."""
    return encrypted_vector.decrypt()