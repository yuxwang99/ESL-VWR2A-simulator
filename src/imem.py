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
        
        assert (n_instr>=0) & (n_instr<RC_NUM_CREG), "Invalid kernel; number of instructions is either negative or too big"

        for index in range(n_instr+1):
            self.rc0_imem.set_word(int(self.instr_df.loc[index+imem_add].RC0,16),index)
            self.rc1_imem.set_word(int(self.instr_df.loc[index+imem_add].RC1,16),index)
            self.rc2_imem.set_word(int(self.instr_df.loc[index+imem_add].RC2,16),index)
            self.rc3_imem.set_word(int(self.instr_df.loc[index+imem_add].RC3,16),index)
            self.lsu_imem.set_word(int(self.instr_df.loc[index+imem_add].LSU,16),index)
            self.lcu_imem.set_word(int(self.instr_df.loc[index+imem_add].LCU,16),index)
            self.mxcu_imem.set_word(int(self.instr_df.loc[index+imem_add].MXCU,16),index)
            
    def get_pos_summary(self, imem_pos=0):
        '''Print what is going on in every specialized slot at a given position in the instruction memory'''
        print("****RC0****")
        self.rc0_imem.get_instruction_info(imem_pos)
        print("****RC1****")
        self.rc1_imem.get_instruction_info(imem_pos)
        print("****RC2****")
        self.rc2_imem.get_instruction_info(imem_pos)
        print("****RC3****")
        self.rc3_imem.get_instruction_info(imem_pos)
        print("****LSU****")
        self.lsu_imem.get_instruction_info(imem_pos)
        print("****LCU****")
        self.lcu_imem.get_instruction_info(imem_pos)
        print("****MXCU****")
        self.mxcu_imem.get_instruction_info(imem_pos)
    
    def get_df(self):
        '''Interate through all kernels and generate a pandas DateFrame with all of the hex instructions, which can then be loaded into a VWR2A testbench'''
        
        df_out = pd.DataFrame(columns=['LCU', 'LSU', 'MXCU', 'RC0', 'RC1', 'RC2', 'RC3', 'KMEM'])
        
        # Fill with default instructions
        default_RC_word = hex(int(RC_IMEM_WORD().get_word(),2))
        default_LSU_word = hex(int(LSU_IMEM_WORD().get_word(),2))
        default_LCU_word = hex(int(LCU_IMEM_WORD().get_word(),2))
        default_MXCU_word = hex(int(MXCU_IMEM_WORD().get_word(),2))
        default_kernel_word = hex(int(KMEM_WORD().get_word(),2))

        df_out.RC0 = np.tile(default_RC_word,IMEM_N_LINES)
        df_out.RC1 = np.tile(default_RC_word,IMEM_N_LINES)
        df_out.RC2 = np.tile(default_RC_word,IMEM_N_LINES)
        df_out.RC3 = np.tile(default_RC_word,IMEM_N_LINES)
        df_out.LSU = np.tile(default_LSU_word,IMEM_N_LINES)
        df_out.LCU = np.tile(default_LCU_word,IMEM_N_LINES)
        df_out.MXCU = np.tile(default_MXCU_word,IMEM_N_LINES)
        
        # For each kernel, populate the DataFrame with hex instructions corresponding to the imem words at the desired positions
        for kernel_idx in range(KER_CONF_N_REG):
            n_instr, imem_add, col, spm_add = self.kmem.get_params(kernel_idx)
            if n_instr > 0:
                df_out.KMEM.iloc[kernel_idx] = self.kmem.get_word_in_hex(kernel_idx)
                for idx in range(n_instr + 1):
                    df_out.RC0.iloc[idx+imem_add] = self.rc0_imem.get_word_in_hex(idx)
                    df_out.RC1.iloc[idx+imem_add] = self.rc1_imem.get_word_in_hex(idx)
                    df_out.RC2.iloc[idx+imem_add] = self.rc2_imem.get_word_in_hex(idx)
                    df_out.RC3.iloc[idx+imem_add] = self.rc3_imem.get_word_in_hex(idx)
                    df_out.LCU.iloc[idx+imem_add] = self.lcu_imem.get_word_in_hex(idx)
                    df_out.LSU.iloc[idx+imem_add] = self.lsu_imem.get_word_in_hex(idx)
                    df_out.MXCU.iloc[idx+imem_add] = self.mxcu_imem.get_word_in_hex(idx)
            else: 
                df_out.KMEM.iloc[kernel_idx] = default_kernel_word
        
        return df_out
        
        
        