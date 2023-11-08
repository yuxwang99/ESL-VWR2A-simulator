"""rc.py: Data structures and objects emulating a Reconfigurable Cell of the VWR2A architecture"""
__author__      = "Lara Orlandic"
__email__       = "lara.orlandic@epfl.ch"

import numpy as np
from enum import Enum

# Local data register (DREG) sizes of specialized slots
RC_NUM_DREG = 2

# Configuration register (CREG) / instruction memory sizes of specialized slots
RC_NUM_CREG = 64

# Widths of instructions of each specialized slot in bits
RC_IMEM_WIDTH = 18

# RC IMEM word decoding
class RC_ALU_OPS(int, Enum):
    '''RC ALU operation codes'''
    NOP = 0
    SADD = 1
    SSUB = 2
    SMUL = 3
    SDIV = 4
    SLL = 5
    SRL = 6
    SRA = 7
    LAND = 8
    LOR = 9
    LXOR = 10
    INB_SF_INA = 11
    INB_ZF_INA = 12
    FXP_MUL = 13
    FXP_DIV = 14

class RC_MUXA_SEL(int, Enum):
    '''Input A to RC ALU'''
    VWR_A = 0
    VWR_B = 1
    VWR_C = 2
    SRF = 3
    R0 = 4
    R1 = 5
    RCT = 6
    RCB = 7
    RCL = 8
    RCR = 9
    ZERO = 10
    ONE = 11
    MAX_INT = 12
    MIN_INT = 13

class RC_MUXB_SEL(int, Enum):
    '''Input B to RC ALU'''
    VWR_A = 0
    VWR_B = 1
    VWR_C = 2
    SRF = 3
    R0 = 4
    R1 = 5
    RCT = 6
    RCB = 7
    RCL = 8
    RCR = 9
    ZERO = 10
    ONE = 11
    MAX_INT = 12
    MIN_INT = 13

class RC_MUXF_SEL(int, Enum):
    '''Select the ALU origin of the data on which to compute flags for SF and ZF operations'''
    OWN = 0
    RCT = 1
    RCB = 2
    RCL = 3
    RCR = 4
    
# RECONFIGURABLE CELL (RC) #

class RC_IMEM:
    '''Instruction memory of the Reconfigurable Cell'''
    def __init__(self):
        self.IMEM = np.zeros(RC_NUM_CREG,dtype="S{0}".format(RC_IMEM_WIDTH))
        # Initialize kernel memory with default word
        default_word = RC_IMEM_WORD()
        for i, instruction in enumerate(self.IMEM):
            self.IMEM[i] = default_word.get_word()
    
    def set_word(self, kmem_word, pos):
        '''Set the IMEM index at integer pos to the binary imem word'''
        self.IMEM[pos] = np.binary_repr(kmem_word,width=RC_IMEM_WIDTH)
    
    def set_params(self, rf_wsel, rf_we, muxf_sel, alu_op, op_mode, muxb_sel, muxa_sel, pos):
        '''Set the IMEM index at integer pos to the configuration parameters.
        See RC_IMEM_WORD initializer for implementation details.
        '''
        imem_word = RC_IMEM_WORD(rf_wsel, rf_we, muxf_sel, alu_op, op_mode, muxb_sel, muxa_sel)
        self.IMEM[pos] = imem_word.get_word()
    
    def get_instruction_info(self, pos):
        '''Print the human-readable instructions of the instruction at position pos in the instruction memory'''
        imem_word = RC_IMEM_WORD()
        imem_word.set_word(self.IMEM[pos])
        rf_wsel, rf_we, muxf_sel, alu_op, op_mode, muxb_sel, muxa_sel = imem_word.decode_word()
        
        
        if op_mode==0:
            precision = "32-bit"
        else:
            precision = "16-bit"
        print("ALU is performing operations with {0} precision".format(precision))
        
        for op in RC_ALU_OPS:
            if op.value == alu_op:
                alu_opcode = op.name
        for sel in RC_MUXA_SEL:
            if sel.value == muxa_sel:
                muxa_res = sel.name
        for sel in RC_MUXB_SEL:
            if sel.value == muxb_sel:
                muxb_res = sel.name
        for sel in RC_MUXF_SEL:
            if sel.value == muxf_sel:
                muxf_res = sel.name
                
        if alu_opcode == RC_ALU_OPS.NOP.name:
            print("No ALU operation")
        elif (alu_opcode == RC_ALU_OPS.INB_SF_INA.name):
            print("Output {0} if sign flag of {1} == 1, else output {2}".format(muxa_res, muxf_res, muxb_res))
        elif (alu_opcode == RC_ALU_OPS.INB_ZF_INA.name):
            print("Output {0} if zero flag of {1} == 1, else output {2}".format(muxa_res, muxf_res, muxb_res))
        else:
            print("Performing ALU operation {0} between operands {1} and {2}".format(alu_opcode, muxa_res, muxb_res))
        
        if rf_we == 1:
            print("Writing ALU result to RC register {0}".format(rf_wsel))
        else:
            print("No RC registers are being written")
        
        
        
    def get_word_in_hex(self, pos):
        '''Get the hexadecimal representation of the word at index pos in the RC config IMEM'''
        return(hex(int(self.IMEM[pos],2)))
        
    
        
class RC_IMEM_WORD:
    def __init__(self, rf_wsel=0, rf_we=0, muxf_sel=0, alu_op=0, op_mode=0, muxb_sel=0, muxa_sel=0):
        '''Generate a binary lsu instruction word from its configuration paramerers:
        
           -   rf_wsel: Select one of eight RC registers to write to
           -   rf_we: Enable writing to aforementioned register
           -   muxf_sel: Select a source for the “flag” parameter that is used to compute the zero and sign flags for some ALU operations
           -   alu_op: Perform one of the ALU operations listed in the RC_ALU_OPS enum
           -   op_mode: Constant 0 for now
           -   muxb_sel: Select input B to ALU (see RC_MUXB_SEL enum for options)
           -   muxa_sel: Select input A to ALU (see RC_MUXA_SEL enum for options)
        
        '''
        self.rf_wsel = np.binary_repr(rf_wsel, width=1)
        self.rf_we = np.binary_repr(rf_we,width=1)
        self.muxf_sel = np.binary_repr(muxf_sel,width=3)
        self.alu_op = np.binary_repr(alu_op,4)
        self.op_mode = np.binary_repr(op_mode,width=1)
        self.muxb_sel = np.binary_repr(muxb_sel,4)
        self.muxa_sel = np.binary_repr(muxa_sel,4)
        self.word = "".join((self.muxa_sel,self.muxb_sel,self.op_mode,self.alu_op,self.muxf_sel,self.rf_we,self.rf_wsel))
    
    def get_word(self):
        return self.word
    
    def set_word(self, word):
        '''Set the binary configuration word of the kernel memory'''
        self.word = word
        self.rf_wsel = word[17:]
        self.rf_we = word[16:17]
        self.muxf_sel = word[13:16]
        self.alu_op = word[9:13]
        self.op_mode = word[8:9]
        self.muxb_sel = word[4:8]
        self.muxa_sel = word[0:4]
        
    
    def decode_word(self):
        '''Get the configuration word parameters from the binary word'''
        rf_wsel = int(self.rf_wsel,2)
        rf_we = int(self.rf_we,2)
        muxf_sel = int(self.muxf_sel,2)
        alu_op = int(self.alu_op,2)
        op_mode = int(self.op_mode,2)
        muxb_sel = int(self.muxb_sel,2)
        muxa_sel = int(self.muxa_sel,2)
        
        
        return rf_wsel, rf_we, muxf_sel, alu_op, op_mode, muxb_sel, muxa_sel