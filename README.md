CompilerError: Failed in nopython mode pipeline (step: native lowering)
Failed in nopython mode pipeline (step: convert to parfors)
Illegal IR, del found at: del $argmax_parallel_impl_v25__v76call_function_32.461

File "..\..\..\..\..\..\ProgramData\Anaconda3\lib\site-packages\numba\parfors\parfor.py", line 183:
def argmax_parallel_impl(in_arr):
    <source elided>
    numba.parfors.parfor.init_prange()
    argmax_checker(len(in_arr))
    ^

During: lowering "id=8[LoopNest(index_variable = parfor_index.347, range = (0, current_indices_size0.320, 1))]{160: <ir.Block at C:\Users\S.T.alu7\AppData\Local\Temp\ipykernel_15712\1664188657.py (33)>, 132: <ir.Block at C:\Users\S.T.alu7\AppData\Local\Temp\ipykernel_15712\1664188657.py (32)>, 230: <ir.Block at C:\Users\S.T.alu7\AppData\Local\Temp\ipykernel_15712\1664188657.py (42)>, 172: <ir.Block at C:\Users\S.T.alu7\AppData\Local\Temp\ipykernel_15712\1664188657.py (34)>, 558: <ir.Block at C:\Users\S.T.alu7\AppData\Local\Temp\ipykernel_15712\1664188657.py (26)>, 560: <ir.Block at C:\Users\S.T.alu7\AppData\Local\Temp\ipykernel_15712\1664188657.py (42)>, 124: <ir.Block at C:\Users\S.T.alu7\AppData\Local\Temp\ipykernel_15712\1664188657.py (32)>, 144: <ir.Block at C:\Users\S.T.alu7\AppData\Local\Temp\ipykernel_15712\1664188657.py (32)>, 212: <ir.Block at C:\Users\S.T.alu7\AppData\Local\Temp\ipykernel_15712\1664188657.py (38)>, 50: <ir.Block at C:\Users\S.T.alu7\AppData\Local\Temp\ipykernel_15712\1664188657.py (18)>, 94: <ir.Block at C:\Users\S.T.alu7\AppData\Local\Temp\ipykernel_15712\1664188657.py (26)>, 92: <ir.Block at C:\Users\S.T.alu7\AppData\Local\Temp\ipykernel_15712\1664188657.py (26)>, 222: <ir.Block at C:\Users\S.T.alu7\AppData\Local\Temp\ipykernel_15712\1664188657.py (41)>}Var(parfor_index.347, 1664188657.py:18)" at C:\Users\S.T.alu7\AppData\Local\Temp\ipykernel_15712\1664188657.py (18)
