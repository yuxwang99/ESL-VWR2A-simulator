"""imem.py: Data structures and objects emulating the Instruction Memory of the VWR2A architecture"""
__author__      = "Lara Orlandic"
__email__       = "lara.orlandic@epfl.ch"

import numpy as np
import pandas as pd
from enum import Enum

from .rc import *
from .mxcu import *
from .lsu import *
from .ker_conf import *
from .lcu import *

# Number of lines in the instruction memory (i.e. max number of instrucitons in all kernels)
IMEM_N_LINES = 512

# INSTRUCTION MEMORY (IMEM) #

class IMEM:
    '''Instruction memory of the CGRA'''
    def __init__(self, instr_df=None):
        # Initialize instruction memories of specialized slots        
        self.rc0_imem = RC_IMEM()
        self.rc1_imem = RC_IMEM()
        self.rc2_imem = RC_IMEM()
        self.rc3_imem = RC_IMEM()
        self.lsu_imem = LSU_IMEM()
        self.lcu_imem = LCU_IMEM()
        self.mxcu_imem = MXCU_IMEM()
        self.kmem = KER_CONF()
        self.instr_df = instr_df
        
        if instr_df is not None:
            # Load kernel configuration into KMEM
            for i in range(KER_CONF_N_REG):
                self.kmem.set_word(int(self.instr_df.loc[i].KMEM,16),i)
    
    def load_kernel(self, kernel_pos=0):
        '''Populate each specialized slot IMEM with the instructions of a given kernel'''
        # Decode kernel instructions
        n_instr, imem_add, col, spm_add = self.kmem.get_params(kernel_pos)
        
        assert (n_instr>0) & (n_instr<RC_NUM_CREG), "Invalid kernel; number of instructions is either negative or too big"

        for index in range(imem_add,imem_add+n_instr+1):
            self.rc0_imem.set_word(int(self.instr_df.loc[index].RC0,16),index)
            self.rc1_imem.set_word(int(self.instr_df.loc[index].RC1,16),index)
            self.rc2_imem.set_word(int(self.instr_df.loc[index].RC2,16),index)
            self.rc3_imem.set_word(int(self.instr_df.loc[index].RC3,16),index)
            self.lsu_imem.set_word(int(self.instr_df.loc[index].LSU,16),index)
            self.lcu_imem.set_word(int(self.instr_df.loc[index].LCU,16),index)
            self.mxcu_imem.set_word(int(self.instr_df.loc[index].MXCU,16),index)
            
    def get_clock_cycle_summary(self, clk_cycle=0):
        print("****RC0****")
        self.rc0_imem.get_instruction_info(clk_cycle)
        print("****RC1****")
        self.rc1_imem.get_instruction_info(clk_cycle)
        print("****RC2****")
        self.rc2_imem.get_instruction_info(clk_cycle)
        print("****RC3****")
        self.rc3_imem.get_instruction_info(clk_cycle)
        print("****LSU****")
        self.lsu_imem.get_instruction_info(clk_cycle)
        print("****LCU****")
        self.lcu_imem.get_instruction_info(clk_cycle)
        print("****MXCU****")
        self.mxcu_imem.get_instruction_info(clk_cycle)
        