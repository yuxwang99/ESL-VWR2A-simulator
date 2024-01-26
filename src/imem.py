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
from .cgra import N_ROW, N_COL

# Number of lines in the instruction memory (i.e. max number of instrucitons in all kernels)
IMEM_N_LINES = 512

# INSTRUCTION MEMORY (IMEM) #

class IMEM:
    '''Instruction memory of the CGRA'''
    def __init__(self, instr_df=None):
        # Initialize instruction memories of specialized slots        
        self.kmem = KER_CONF()
        self.instr_df = instr_df
        
        # Create IMEM objects for each column
        for col_num in range(N_COL):
            for rc_num in range(N_ROW):
                setattr(self, "rc{0}_imem_col{1}".format(rc_num, col_num), RC_IMEM())
            setattr(self, "lsu_imem_col{0}".format(col_num), LSU_IMEM())
            setattr(self, "lcu_imem_col{0}".format(col_num), LCU_IMEM())
            setattr(self, "mxcu_imem_col{0}".format(col_num), MXCU_IMEM())
        
        if instr_df is not None:
            # Load kernel configuration into KMEM
            for i in range(1,KER_CONF_N_REG):
                self.kmem.set_word(int(self.instr_df.loc[i].KMEM,16),i)
    
    def load_kernel(self, kernel_pos=0):
        '''Populate each specialized slot IMEM with the instructions of a given kernel'''
        # Decode kernel instructions
        n_instr, imem_add, col, spm_add = self.kmem.get_params(kernel_pos)
        
        assert (n_instr>=0) & (n_instr<RC_NUM_CREG), "Invalid kernel; number of instructions is either negative or too big"
        assert (col>0), "The column attribute must be one-hot encoded"
        
        if col == 1:
            active_cols = [0]
        elif col == 2:
            active_cols = [1]
        elif col == 3:
            active_cols = [0,1]
        
        if self.instr_df is not None:
            # Populate instruction memory
            for index in range(n_instr+1):
                getattr(self, "rc0_imem_col{0}".format(active_cols[0])).set_word(int(self.instr_df.loc[index+imem_add].RC0,16),index)
                getattr(self, "rc1_imem_col{0}".format(active_cols[0])).set_word(int(self.instr_df.loc[index+imem_add].RC1,16),index)
                getattr(self, "rc2_imem_col{0}".format(active_cols[0])).set_word(int(self.instr_df.loc[index+imem_add].RC2,16),index)
                getattr(self, "rc3_imem_col{0}".format(active_cols[0])).set_word(int(self.instr_df.loc[index+imem_add].RC3,16),index)
                getattr(self, "lsu_imem_col{0}".format(active_cols[0])).set_word(int(self.instr_df.loc[index+imem_add].LSU,16),index)
                getattr(self, "lcu_imem_col{0}".format(active_cols[0])).set_word(int(self.instr_df.loc[index+imem_add].LCU,16),index)
                getattr(self, "mxcu_imem_col{0}".format(active_cols[0])).set_word(int(self.instr_df.loc[index+imem_add].MXCU,16),index)
            if len(active_cols)>1: #Two-column kernel
                for index in range(n_instr+1,2*(n_instr+1)):
                    self.rc0_imem_col1.set_word(int(self.instr_df.loc[index+imem_add].RC0,16),index-n_instr)
                    self.rc1_imem_col1.set_word(int(self.instr_df.loc[index+imem_add].RC1,16),index-n_instr)
                    self.rc2_imem_col1.set_word(int(self.instr_df.loc[index+imem_add].RC2,16),index-n_instr)
                    self.rc3_imem_col1.set_word(int(self.instr_df.loc[index+imem_add].RC3,16),index-n_instr)
                    self.lsu_imem_col1.set_word(int(self.instr_df.loc[index+imem_add].LSU,16),index-n_instr)
                    self.lcu_imem_col1.set_word(int(self.instr_df.loc[index+imem_add].LCU,16),index-n_instr)
                    self.mxcu_imem_col1.set_word(int(self.instr_df.loc[index+imem_add].MXCU,16),index-n_instr)
            
    def get_pos_summary(self, imem_pos=0, col_index=0):
        '''Print what is going on in every specialized slot at a given position in the instruction memory of a specified column'''
        for i in range(N_ROW):
            print("****RC{0}****".format(i))
            getattr(self,"rc{0}_imem_col{1}".format(i, col_index)).get_instruction_info(imem_pos)
        print("****LSU****")
        getattr(self,"lsu_imem_col{0}".format(col_index)).get_instruction_info(imem_pos)
        print("****LCU****")
        getattr(self,"lcu_imem_col{0}".format(col_index)).get_instruction_info(imem_pos)
        print("****MXCU****")
        getattr(self,"mxcu_imem_col{0}".format(col_index)).get_instruction_info(imem_pos)
    
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
            if col > 0:
                self.load_kernel(kernel_idx)
                df_out.KMEM.iloc[kernel_idx] = self.kmem.get_word_in_hex(kernel_idx)
                
                if col == 1:
                    active_cols = [0]
                elif col == 2:
                    active_cols = [1]
                elif col == 3:
                    active_cols = [0,1]
                
                for idx in range(n_instr + 1):
                    df_out.RC0.iloc[idx+imem_add] = getattr(self, "rc0_imem_col{0}".format(active_cols[0])).get_word_in_hex(idx)
                    df_out.RC1.iloc[idx+imem_add] = getattr(self, "rc1_imem_col{0}".format(active_cols[0])).get_word_in_hex(idx)
                    df_out.RC2.iloc[idx+imem_add] = getattr(self, "rc2_imem_col{0}".format(active_cols[0])).get_word_in_hex(idx)
                    df_out.RC3.iloc[idx+imem_add] = getattr(self, "rc3_imem_col{0}".format(active_cols[0])).get_word_in_hex(idx)
                    df_out.LCU.iloc[idx+imem_add] = getattr(self, "lcu_imem_col{0}".format(active_cols[0])).get_word_in_hex(idx)
                    df_out.LSU.iloc[idx+imem_add] = getattr(self, "lsu_imem_col{0}".format(active_cols[0])).get_word_in_hex(idx)
                    df_out.MXCU.iloc[idx+imem_add] = getattr(self, "mxcu_imem_col{0}".format(active_cols[0])).get_word_in_hex(idx)
                if len(active_cols)>1: #2-column kernels
                    for idx in range(n_instr+1,2*(n_instr+1)):
                        df_out.RC0.iloc[idx+imem_add] = self.rc0_imem_col1.get_word_in_hex(idx-n_instr)
                        df_out.RC1.iloc[idx+imem_add] = self.rc1_imem_col1.get_word_in_hex(idx-n_instr)
                        df_out.RC2.iloc[idx+imem_add] = self.rc2_imem_col1.get_word_in_hex(idx-n_instr)
                        df_out.RC3.iloc[idx+imem_add] = self.rc3_imem_col1.get_word_in_hex(idx-n_instr)
                        df_out.LCU.iloc[idx+imem_add] = self.lcu_imem_col1.get_word_in_hex(idx-n_instr)
                        df_out.LSU.iloc[idx+imem_add] = self.lsu_imem_col1.get_word_in_hex(idx-n_instr)
                        df_out.MXCU.iloc[idx+imem_add] = self.mxcu_imem_col1.get_word_in_hex(idx-n_instr)
            else: 
                df_out.KMEM.iloc[kernel_idx] = default_kernel_word
        
        return df_out
        
        
        