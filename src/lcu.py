"""lcu.py: Data structures and objects emulating the Loop Control Unit of the VWR2A architecture"""
__author__      = "Lara Orlandic"
__email__       = "lara.orlandic@epfl.ch"

import numpy as np
from enum import Enum

# Local data register (DREG) sizes of specialized slots
LCU_NUM_DREG = 4

# Configuration register (CREG) / instruction memory sizes of specialized slots
LCU_NUM_CREG = 64

# Widths of instructions of each specialized slot in bits
LCU_IMEM_WIDTH = 20

# LCU IMEM word decoding
class LCU_ALU_OPS(int, Enum):
    '''LCU ALU operation codes'''
    NOP = 0
    SADD = 1
    SSUB = 2
    SLL = 3
    SRL = 4
    SRA = 5
    LAND = 6
    LOR = 7
    LXOR = 8
    BEQ = 9
    BNE = 10
    BGEPD = 11
    BLT = 12
    JUMP = 13
    EXIT = 14

class LCU_MUXA_SEL(int, Enum):
    '''Input A to LCU ALU'''
    R0 = 0
    R1 = 1
    R2 = 2
    R3 = 3
    SRF = 4
    LAST = 5
    ZERO = 6
    IMM = 7

class LCU_MUXB_SEL(int, Enum):
    '''Input B to LCU ALU'''
    R0 = 0
    R1 = 1
    R2 = 2
    R3 = 3
    SRF = 4
    LAST = 5
    ZERO = 6
    ONE = 7
    
# LOOP CONTROL UNIT (LCU) #

class LCU_IMEM:
    '''Instruction memory of the Loop Control Unit'''
    def __init__(self):
        self.IMEM = np.zeros(LCU_NUM_CREG,dtype="S{0}".format(LCU_IMEM_WIDTH))
        # Initialize kernel memory with zeros
        for i, instruction in enumerate(self.IMEM):
            self.IMEM[i] = np.binary_repr(0,width=LCU_IMEM_WIDTH)
    
    def set_kernel_word(self, kmem_word, pos):
        '''Set the IMEM index at integer pos to the binary imem word'''
        self.IMEM[pos] = np.binary_repr(kmem_word,width=LCU_IMEM_WIDTH)
    
    def set_kernel_params(self, imm, rf_wsel, rf_we, alu_op, br_mode, muxb_sel, muxa_sel, pos):
        '''Set the IMEM index at integer pos to the configuration parameters.
        See LCU_IMEM_WORD initializer for implementation details.
        '''
        imem_word = LCU_IMEM_WORD(imm, rf_wsel, rf_we, alu_op, br_mode, muxb_sel, muxa_sel)
        self.IMEM[pos] = imem_word.get_word()
    
    def get_instruction_info(self, pos):
        '''Print the human-readable instructions of the instruction at position pos in the instruction memory'''
        imem_word = LCU_IMEM_WORD()
        imem_word.set_word(self.IMEM[pos])
        imm, rf_wsel, rf_we, alu_op, br_mode, muxb_sel, muxa_sel = imem_word.decode_word()
        
        print("Immediate value: {0}".format(imm))
        
        if br_mode == 1:
            print ("LCU is in RC data control mode")
        else: 
            print ("LCU is in loop control mode")
            
        for op in LCU_ALU_OPS:
            if op.value == alu_op:
                alu_opcode = op.name
        for sel in LCU_MUXA_SEL:
            if sel.value == muxa_sel:
                muxa_res = sel.name
        for sel in LCU_MUXB_SEL:
            if sel.value == muxb_sel:
                muxb_res = sel.name
        if alu_op == 0: #NOP
            print("No LCU ALU Operation is performed")
        elif alu_op == 9: #BEQ
            print("If {0} and {1} are equal, branch to the immediate value {2}".format(muxa_res, muxb_res, imm))
        elif alu_op == 10: #BNE
            print("If {0} and {1} are NOT equal, branch to the immediate value {2}".format(muxa_res, muxb_res, imm))
        elif alu_op == 11: #BGEPD
            print("If {0}-1 is greater than or equal to {1}, branch to the immediate value {2}".format(muxa_res, muxb_res, imm))
        elif alu_op == 12: #BLT
            print("If {0} is less than {1}, branch to the immediate value {2}".format(muxa_res, muxb_res, imm))
        elif alu_op == 13: #JUMP
            print("Jump to address {0} + {1}".format(muxa_res, muxb_res))
        elif alu_op == 14: #EXIT
            print("Exiting out of kernel")
        else:
            print("Performing ALU operation {0} between operands {1} and {2}".format(alu_opcode, muxa_res, muxb_res))
        
        if rf_we == 1:
            print("Writing ALU result to LCU register {0}".format(rf_wsel))
        else:
            print("No LCU registers are being written")

        
        
    def get_word_in_hex(self, pos):
        '''Get the hexadecimal representation of the word at index pos in the LCU config IMEM'''
        return(hex(int(self.IMEM[pos],2)))
        
    
        
class LCU_IMEM_WORD:
    def __init__(self, imm=0, rf_wsel=0, rf_we=0, alu_op=0, br_mode=0, muxb_sel=0, muxa_sel=0):
        '''Generate a binary lcu instruction word from its configuration paramerers:
        
           -   imm: Immediate value to use for ALU operations or address to branch to
           -   rf_wsel: Select one of four LCU registers to write to
           -   rf_we: Enable writing to aforementioned register
           -   alu_op: Perform one of the ALU operations listed in the LCU_ALU_OPS enum
           -   br_mode: Control program counter (0) or RC datapath (1)
           -   muxb_sel: Select input B to ALU (see LCU_MUXB_SEL enum for options)
           -   muxa_sel: Select input A to ALU (see LCU_MUXA_SEL enum for options)
        
        '''
        self.imm = np.binary_repr(imm, width=6)
        self.rf_wsel = np.binary_repr(rf_wsel, width=2)
        self.rf_we = np.binary_repr(rf_we,width=1)
        self.alu_op = np.binary_repr(alu_op,4)
        self.br_mode = np.binary_repr(br_mode,1)
        self.muxb_sel = np.binary_repr(muxb_sel,3)
        self.muxa_sel = np.binary_repr(muxa_sel,3)
        self.word = "".join((self.muxa_sel,self.muxb_sel,self.br_mode,self.alu_op,self.rf_we,self.rf_wsel,self.imm))
    
    def get_word(self):
        return self.word
    
    def set_word(self, word):
        '''Set the binary configuration word of the kernel memory'''
        self.word = word
        self.imm = word[14:]
        self.rf_wsel = word[12:14]
        self.rf_we = word[11:12]
        self.alu_op = word[7:11]
        self.br_mode = word[6:7]
        self.muxb_sel = word[3:6]
        self.muxa_sel = word[0:3]
        
    
    def decode_word(self):
        '''Get the configuration word parameters from the binary word'''
        imm = int(self.imm,2)
        rf_wsel = int(self.rf_wsel,2)
        rf_we = int(self.rf_we,2)
        alu_op = int(self.alu_op,2)
        br_mode = int(self.br_mode,2)
        muxb_sel = int(self.muxb_sel,2)
        muxa_sel = int(self.muxa_sel,2)
        
        return imm, rf_wsel, rf_we, alu_op, br_mode, muxb_sel, muxa_sel