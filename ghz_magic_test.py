from distill_circuit import *
# import numpy as np

zero_state = statevector.Statevector(np.array([1, 0]))
one_state = statevector.Statevector(np.array([0, 1]))
t0_5 = tensor_n(t0, 5)
t1_5 = tensor_n(t1, 5)
eps = .5
alpha = .88807
beta = np.sqrt(1 - alpha**2)
ghz_phase = alpha * tensor_n(zero_state, 5) +  np.exp(1j*np.pi/4) * beta * tensor_n(one_state, 5)
single_qubit = qi.partial_trace(ghz_phase, range(4))

print("Bravyi Kitaev on single qubits obtained from ghz phase states")
# print(single_qubit)
print(magic_fidelity(single_qubit, 0))
bk_input = tensor_n(single_qubit, 5)
full_output = distill(bk_input)
output_qubit = qi.partial_trace(full_output, range(4))
print(magic_fidelity(output_qubit, 0))

print("Bravyi Kitaev on ghz phase states")
ghz_output = distill(ghz_phase)
ghz_out_qubit =  qi.partial_trace(ghz_output, range(4))
print(magic_fidelity(ghz_out_qubit, 1))

print("Bravyi Kitaev^2 on ghz phase states")
ghz_2 = tensor_n(ghz_out_qubit, 5)
ghz_2_output = distill(ghz_2)
ghz_2_out_qubit =  qi.partial_trace(ghz_2_output, range(4))
print(magic_fidelity(ghz_2_out_qubit, 1))

rdm = ghz_2_out_qubit

repeat_distill(rdm, 10)
