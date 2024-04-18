"""lsu.py: Data structures and objects emulating the Load Store Unit of the VWR2A architecture"""
__author__      = "Lara Orlandic"
__email__       = "lara.orlandic@epfl.ch"

import numpy as np
from enum import Enum

# Local data register (DREG) sizes of specialized slots
LSU_NUM_DREG = 8

# Configuration register (CREG) / instruction memory sizes of specialized slots
LSU_NUM_CREG = 64

# Widths of instructions of each specialized slot in bits
LSU_IMEM_WIDTH = 20

# LSU IMEM word decoding
class LSU_ALU_OPS(int, Enum):
    '''LSU ALU operation codes'''
    LAND = 0
    LOR = 1
    LXOR = 2
    SADD = 3
    SSUB = 4
    SLL = 5
    SRL = 6
    BITREV = 7

class LSU_MUXA_SEL(int, Enum):
    '''Input A to LSU ALU'''
    R0 = 0
    R1 = 1
    R2 = 2
    R3 = 3
    R4 = 4
    R5 = 5
    R6 = 6
    R7 = 7
    SRF = 8
    ZERO = 9
    ONE = 10
    TWO = 11

class LSU_MUXB_SEL(int, Enum):
    '''Input B to LSU ALU'''
    R0 = 0
    R1 = 1
    R2 = 2
    R3 = 3
    R4 = 4
    R5 = 5
    R6 = 6
    R7 = 7
    SRF = 8
    ZERO = 9
    ONE = 10
    TWO = 11

class LSU_OP_MODE(int, Enum):
    '''Select whether the LSU is interfacing with the SPM or shuffling VWR data'''
    NOP = 0
    LOAD = 1
    STORE = 2
    SHUFFLE = 3

class LSU_VWR_SEL(int, Enum):
    '''When the LSU OP MODE is in LOAD/STORE, choose which element to load or store from'''
    VWR_A = 0
    VWR_B = 1
    VWR_C = 2
    SRF = 3
    
class SHUFFLE_SEL(int, Enum):
    '''When the LSU OP MODE is in SHUFFLE, choose how to shuffle VWRs A and B into VWR C'''
    INTERLEAVE_UPPER = 0
    INTERLEAVE_LOWER = 1
    EVEN_INDICES = 2
    ODD_INDICES = 3
    CONCAT_BITREV_UPPER = 4
    CONCAT_BITREV_LOWER = 5
    CONCAT_SLICE_CIRCULAR_SHIFT_UPPER = 6
    CONCAT_SLICE_CIRCULAR_SHIFT_LOWER = 7
    
# LOAD STORE UNIT (LSU) #

class LSU_IMEM:
    '''Instruction memory of the Load Store Unit'''
    def __init__(self):
        self.IMEM = np.zeros(LSU_NUM_CREG,dtype="S{0}".format(LSU_IMEM_WIDTH))
        # Initialize kernel memory with default instruction
        default_word = LSU_IMEM_WORD()
        for i, instruction in enumerate(self.IMEM):
            self.IMEM[i] = default_word.get_word()
    
    def set_word(self, kmem_word, pos):
        '''Set the IMEM index at integer pos to the binary imem word'''
        self.IMEM[pos] = np.binary_repr(kmem_word,width=LSU_IMEM_WIDTH)
    
    def set_params(self, rf_wsel=0, rf_we=0, alu_op=LSU_ALU_OPS.LAND, muxb_sel=LSU_MUXB_SEL.ZERO, muxa_sel=LSU_MUXA_SEL.ZERO, vwr_shuf_op=LSU_VWR_SEL.VWR_A, vwr_shuf_sel=LSU_OP_MODE.NOP, pos=0):
        '''Set the IMEM index at integer pos to the configuration parameters.
        See LSU_IMEM_WORD initializer for implementation details.
        '''
        imem_word = LSU_IMEM_WORD(rf_wsel, rf_we, alu_op, muxb_sel, muxa_sel, vwr_shuf_op, vwr_shuf_sel)
        self.IMEM[pos] = imem_word.get_word()
    
    def get_instruction_info(self, pos):
        '''Print the human-readable instructions of the instruction at position pos in the instruction memory'''
        imem_word = LSU_IMEM_WORD()
        imem_word.set_word(self.IMEM[pos])
        rf_wsel, rf_we, alu_op, muxb_sel, muxa_sel, vwr_shuf_op, vwr_shuf_sel = imem_word.decode_word()
        
        print(f"{pos}: ==============================")
        # See if we are performing a load/store or a shuffle
        for op in LSU_OP_MODE:
            if op.value == vwr_shuf_sel:
                lsu_mode = op.name
        
        # If we are loading/storing ...
        if ((lsu_mode == LSU_OP_MODE.STORE.name)|(lsu_mode == LSU_OP_MODE.LOAD.name)):
            #... which register are we using?
            for reg in LSU_VWR_SEL:
                if reg.value == vwr_shuf_op:
                    register = reg.name
            if (lsu_mode == LSU_OP_MODE.LOAD.name):
                preposition1 = "from"
                preposition2 = "to"
            else:
                preposition1 = "to"
                preposition2 = "from"
            print("Performing {0} {1} SPM {2} {3}".format(lsu_mode, preposition1, preposition2, register))
        # If we are shuffling VWRs A and B into C ...
        elif (lsu_mode == LSU_OP_MODE.SHUFFLE.name):
            #... which one are we using?
            for shuf in SHUFFLE_SEL:
                if shuf.value == vwr_shuf_op:
                    shuffle_type = shuf.name
            print("Shuffling VWR A and B data into VWR C using operation {0}".format(shuffle_type))
        else: # NOP
            print("No loading, storing, or shuffling taking place")
        
        for op in LSU_ALU_OPS:
            if op.value == alu_op:
                alu_opcode = op.name
        for sel in LSU_MUXA_SEL:
            if sel.value == muxa_sel:
                muxa_res = sel.name
        for sel in LSU_MUXB_SEL:
            if sel.value == muxb_sel:
                muxb_res = sel.name
        
        print("Performing ALU operation {0} between operands {1} and {2}".format(alu_opcode, muxa_res, muxb_res))
        
        if rf_we == 1:
            print("Writing ALU result to LSU register {0}".format(rf_wsel))
        else:
            print("No LSU registers are being written")
        
        
    def get_word_in_hex(self, pos):
        '''Get the hexadecimal representation of the word at index pos in the LCU config IMEM'''
        return(hex(int(self.IMEM[pos],2)))
        
    
        
class LSU_IMEM_WORD:
    def __init__(self, rf_wsel=0, rf_we=0, alu_op=LSU_ALU_OPS.LAND, muxb_sel=LSU_MUXB_SEL.ZERO, muxa_sel=LSU_MUXA_SEL.ZERO, vwr_shuf_op=LSU_VWR_SEL.VWR_A, vwr_shuf_sel=LSU_OP_MODE.NOP):
        '''Generate a binary lsu instruction word from its configuration paramerers:
        
           -   rf_wsel: Select one of eight LSU registers to write to
           -   rf_we: Enable writing to aforementioned register
           -   alu_op: Perform one of the ALU operations listed in the LSU_ALU_OPS enum
           -   muxb_sel: Select input B to ALU (see LSU_MUXB_SEL enum for options)
           -   muxa_sel: Select input A to ALU (see LSU_MUXA_SEL enum for options)
           -   vwr_shuf_op: Decide which VWR to load/store to (LSU_VWR_SEL) or which shuffle operation to perform (SHUFFLE_SEL)
           -   vwr_shuf_sel: Decide whether to use LSU for SPM communication or data shuffling (see LSU_OP_MODE enum for options)
        
        '''
        self.rf_wsel = np.binary_repr(rf_wsel, width=3)
        self.rf_we = np.binary_repr(rf_we,width=1)
        self.alu_op = np.binary_repr(alu_op,3)
        self.muxb_sel = np.binary_repr(muxb_sel,4)
        self.muxa_sel = np.binary_repr(muxa_sel,4)
        self.vwr_shuf_op = np.binary_repr(vwr_shuf_op,3)
        self.vwr_shuf_sel = np.binary_repr(vwr_shuf_sel,2)
        self.word = "".join((self.vwr_shuf_sel,self.vwr_shuf_op,self.muxa_sel,self.muxb_sel,self.alu_op,self.rf_we,self.rf_wsel))
    
    def get_word(self):
        return self.word
    
    def set_word(self, word):
        '''Set the binary configuration word of the kernel memory'''
        self.word = word
        self.rf_wsel = word[17:]
        self.rf_we = word[16:17]
        self.alu_op = word[13:16]
        self.muxb_sel = word[9:13]
        self.muxa_sel = word[5:9]
        self.vwr_shuf_op = word[2:5]
        self.vwr_shuf_sel = word[0:2]
        
    
    def decode_word(self):
        '''Get the configuration word parameters from the binary word'''
        rf_wsel = int(self.rf_wsel,2)
        rf_we = int(self.rf_we,2)
        alu_op = int(self.alu_op,2)
        muxb_sel = int(self.muxb_sel,2)
        muxa_sel = int(self.muxa_sel,2)
        vwr_shuf_op = int(self.vwr_shuf_op,2)
        vwr_shuf_sel = int(self.vwr_shuf_sel,2)
        
        
        return rf_wsel, rf_we, alu_op, muxb_sel, muxa_sel, vwr_shuf_op, vwr_shuf_sel