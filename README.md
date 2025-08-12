RuntimeError                              Traceback (most recent call last)
Cell In[42], line 1
----> 1 get_ipython().run_cell_magic('time', '', "asyncio.run(process_all(calls_df['masked_text'].head()))\n")

File ~/miniconda3/envs/coa_ai/lib/python3.12/site-packages/IPython/core/interactiveshell.py:2565, in InteractiveShell.run_cell_magic(self, magic_name, line, cell)
   2563 with self.builtin_trap:
   2564     args = (magic_arg_s, cell)
-> 2565     result = fn(*args, **kwargs)
   2567 # The code below prevents the output from being displayed
   2568 # when using magics with decorator @output_can_be_silenced
   2569 # when the last Python token in the expression is a ';'.
   2570 if getattr(fn, magic.MAGIC_OUTPUT_CAN_BE_SILENCED, False):

File ~/miniconda3/envs/coa_ai/lib/python3.12/site-packages/IPython/core/magics/execution.py:1470, in ExecutionMagics.time(self, line, cell, local_ns)
   1468 if interrupt_occured:
   1469     if exit_on_interrupt and captured_exception:
-> 1470         raise captured_exception
   1471     return
   1472 return out

File ~/miniconda3/envs/coa_ai/lib/python3.12/site-packages/IPython/core/magics/execution.py:1420, in ExecutionMagics.time(self, line, cell, local_ns)
   1418 st = clock2()
   1419 try:
-> 1420     out = eval(code, glob, local_ns)
   1421 except KeyboardInterrupt as e:
   1422     captured_exception = e

File <timed eval>:1

File ~/miniconda3/envs/coa_ai/lib/python3.12/asyncio/runners.py:190, in run(main, debug, loop_factory)
    161 """Execute the coroutine and return the result.
    162 
    163 This function runs the passed coroutine, taking care of
   (...)    186     asyncio.run(main())
    187 """
    188 if events._get_running_loop() is not None:
    189     # fail fast with short traceback
--> 190     raise RuntimeError(
    191         "asyncio.run() cannot be called from a running event loop")
    193 with Runner(debug=debug, loop_factory=loop_factory) as runner:
    194     return runner.run(main)

RuntimeError: asyncio.run() cannot be called from a running event loop)
