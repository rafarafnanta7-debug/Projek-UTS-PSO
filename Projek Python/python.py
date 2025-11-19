import numpy as np

# -----------------------------------
# Fungsi bantu
# -----------------------------------

def binary_to_decimal(binary_array):
    """Konversi array biner (setiap baris 4 bit) menjadi bilangan desimal."""
    return np.array([b[0]*8 + b[1]*4 + b[2]*2 + b[3]*1 for b in binary_array])

def fitness_function(x):
    """Fungsi fitness (x - 5)^2 + 10."""
    return (x - 5)**2 + 10

def sigmoid(v):
    """Fungsi sigmoid."""
    return 1 / (1 + np.exp(-v))

# -----------------------------------
# Inisialisasi (sesuai soal)
# -----------------------------------

X = np.array([
    [0, 0, 1, 0],  # -> 2
    [1, 0, 0, 0],  # -> 8
    [1, 1, 0, 0]   # -> 12
], dtype=int)

V = np.zeros_like(X, dtype=float)

# Parameter PSO
w = 0.5
c1 = 1
c2 = 1

# r1, r2, random_check sesuai yang kamu tetapkan
r1 = np.array([0.2, 0.4, 0.6, 0.8])
r2 = np.array([0.9, 0.7, 0.5, 0.3])
random_check = np.array([0.7, 0.2, 0.8, 0.4])

# Hitung fitness awal
fitness = fitness_function(binary_to_decimal(X))

# Set Pbest dan Gbest
Pbest = X.copy()
Pbest_fitness = fitness.copy()

gbest_index = np.argmax(Pbest_fitness)
Gbest = Pbest[gbest_index].copy()
Gbest_fitness = Pbest_fitness[gbest_index]

# -----------------------------------
# ITERASI PSO (2 iterasi sesuai penjelasan)
# -----------------------------------

for t in range(2):
    print("\n================ ITERASI", t+1, "===============")

    for i in range(len(X)):
        # Hitung cognitive & social
        diff_p = Pbest[i] - X[i]
        diff_g = Gbest - X[i]

        cognitive = c1 * r1 * diff_p
        social = c2 * r2 * diff_g

        # Update kecepatan
        V[i] = w * V[i] + cognitive + social

        # Hitung sigmoid
        S = sigmoid(V[i])

        # Update posisi biner berdasarkan random_check
        X[i] = (random_check < S).astype(int)

        print(f"\nPartikel {i}:")
        print(" diff_p       =", diff_p)
        print(" diff_g       =", diff_g)
        print(" cognitive    =", cognitive)
        print(" social       =", social)
        print(" V baru       =", V[i])
        print(" sigmoid(V)   =", S)
        print(" X baru       =", X[i])

    # Hitung fitness baru
    fitness = fitness_function(binary_to_decimal(X))
    print("\nFitness =", fitness)

    # Update Pbest
    for i in range(len(X)):
        if fitness[i] > Pbest_fitness[i]:
            Pbest[i] = X[i].copy()
            Pbest_fitness[i] = fitness[i]

    # Update Gbest
    gbest_index = np.argmax(Pbest_fitness)
    Gbest = Pbest[gbest_index].copy()
    Gbest_fitness = Pbest_fitness[gbest_index]

    print("\nPbest fitness =", Pbest_fitness)
    print("Gbest posisi  =", Gbest)
    print("Gbest fitness =", Gbest_fitness)

# -----------------------------------
# HASIL AKHIR
# -----------------------------------

print("\n=== HASIL AKHIR ===")
print("Gbest (biner)   =", Gbest)
print("Gbest (desimal) =", int(Gbest[0]*8 + Gbest[1]*4 + Gbest[2]*2 + Gbest[3]*1))
print("Fitness maksimum =", Gbest_fitness)