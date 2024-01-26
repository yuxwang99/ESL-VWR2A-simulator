"""mxcu.py: Data structures and objects emulating the Multiplexer Control Unit of the VWR2A architecture"""
__author__      = "Lara Orlandic"
__email__       = "lara.orlandic@epfl.ch"

import numpy as np
from enum import Enum

# Local data register (DREG) sizes of specialized slots
MXCU_NUM_DREG = 8

# Configuration register (CREG) / instruction memory sizes of specialized slots
MXCU_NUM_CREG = 64

# Widths of instructions of each specialized slot in bits
MXCU_IMEM_WIDTH = 27

# MXCU IMEM word decoding
class MXCU_ALU_OPS(int, Enum):
    '''MXCU ALU operation codes'''
    NOP = 0
    SADD = 1
    SSUB = 2
    SLL = 3
    SRL = 4
    LAND = 5
    LOR = 6
    LXOR = 7

class MXCU_MUXA_SEL(int, Enum):
    '''Input A to MXCU ALU'''
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
    HALF_VWR_SIZE = 12
    LAST_VWR = 13

class MXCU_MUXB_SEL(int, Enum):
    '''Input B to MXCU ALU'''
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
    HALF_VWR_SIZE = 12
    LAST_VWR = 13

class ALU_SRF_WRITE(int, Enum):
    '''Select which specialized slot's ALU output gets written to the chosen SRF register'''
    LCU = 0
    RC0 = 1
    MXCU = 2
    LSU = 3

class MXCU_VWR_SEL(int, Enum):
    '''Choose which VWR to write to'''
    VWR_A = 0
    VWR_B = 1
    VWR_C = 2
    
    
# MULTIPLEXER CONTROL UNIT (MXCU) #

class MXCU_IMEM:
    '''Instruction memory of the Multiplexer control unit'''
    def __init__(self):
        self.IMEM = np.zeros(MXCU_NUM_CREG,dtype="S{0}".format(MXCU_IMEM_WIDTH))
        # Initialize kernel memory with default word
        default_word = MXCU_IMEM_WORD()
        for i, instruction in enumerate(self.IMEM):
            self.IMEM[i] = default_word.get_word()
    
    def set_word(self, kmem_word, pos):
        '''Set the IMEM index at integer pos to the binary imem word'''
        self.IMEM[pos] = np.binary_repr(kmem_word,width=MXCU_IMEM_WIDTH)
    
    def set_params(self, vwr_row_we=[0,0,0,0], vwr_sel=MXCU_VWR_SEL.VWR_A, srf_sel=0, alu_srf_write=ALU_SRF_WRITE.LCU, srf_we=0, rf_wsel=0, rf_we=0, alu_op=MXCU_ALU_OPS.NOP, muxb_sel=MXCU_MUXB_SEL.R0, muxa_sel=MXCU_MUXA_SEL.R0, pos=0):
        '''Set the IMEM index at integer pos to the configuration parameters.
        See MXCU_IMEM_WORD initializer for implementation details.
        NOTE: vwr_row_we should be an 4-element array of bool/int values representing a one-hot vector of row write enable bits
        '''
        #Convert one-hot array of int/bool to binary
        imem_word = MXCU_IMEM_WORD(vwr_row_we, vwr_sel, srf_sel, alu_srf_write, srf_we, rf_wsel, rf_we, alu_op, muxb_sel, muxa_sel)
        self.IMEM[pos] = imem_word.get_word()
    
    def get_instruction_info(self, pos):
        '''Print the human-readable instructions of the instruction at position pos in the instruction memory'''
        imem_word = MXCU_IMEM_WORD()
        imem_word.set_word(self.IMEM[pos])
        vwr_row_we, vwr_sel, srf_sel, alu_srf_write, srf_we, rf_wsel, rf_we, alu_op, muxb_sel, muxa_sel = imem_word.decode_word()
        for vwr in MXCU_VWR_SEL:
            if vwr.value == vwr_sel:
                selected_vwr = vwr.name
        
        indices_of_written_rows = np.where(vwr_row_we[::-1])[0]
        if len(indices_of_written_rows)>0:
            print("Writing to VWR rows {0} of {1}".format(indices_of_written_rows, selected_vwr))
        else:
            print("Not writing to VWRs")
        
        if srf_we == 1:
            for alu_res in ALU_SRF_WRITE:
                if alu_res.value == alu_srf_write:
                    spec_slot = alu_res.name
            print("Writing from {0} ALU to SRF register {1}".format(spec_slot, srf_sel))
        else:
            print("Reading from SRF index {0}".format(srf_sel))
        
        for op in MXCU_ALU_OPS:
            if op.value == alu_op:
                alu_opcode = op.name
        for sel in MXCU_MUXA_SEL:
            if sel.value == muxa_sel:
                muxa_res = sel.name
        for sel in MXCU_MUXB_SEL:
            if sel.value == muxb_sel:
                muxb_res = sel.name
        if alu_opcode == MXCU_ALU_OPS.NOP:
            print("No ALU operation")
        else:
            print("Performing ALU operation {0} between operands {1} and {2}".format(alu_opcode, muxa_res, muxb_res))
        if rf_we == 1:
            print("Writing ALU result to MXCU register {0}".format(rf_wsel))
        else:
            print("No MXCU registers are being written")
        
        
        
    def get_word_in_hex(self, pos):
        '''Get the hexadecimal representation of the word at index pos in the LCU config IMEM'''
        return(hex(int(self.IMEM[pos],2)))
        
    
        
class MXCU_IMEM_WORD:
    def __init__(self, vwr_row_we=[0,0,0,0], vwr_sel=MXCU_VWR_SEL.VWR_A, srf_sel=0, alu_srf_write=ALU_SRF_WRITE.LCU, srf_we=0, rf_wsel=0, rf_we=0, alu_op=MXCU_ALU_OPS.NOP, muxb_sel=MXCU_MUXB_SEL.R0, muxa_sel=MXCU_MUXA_SEL.R0):
        '''Generate a binary mxcu instruction word from its configuration paramerers:
        
           -   vwr_row_we: One-hot encoded write enable to the 4 rows (also known as slices) of the VWR.
           -   vwr_sel: Select which VWR to write to (see MXCU_VWR_SEL for options)
           -   srf_sel: Select one of 8 SRF registers to read/write to
           -   alu_srf_write: Decide which specialized slot ALU result to write to selected SRF register (see ALU_SRF_WRITE enum)
           -   srf_we: Write enable to the SRF
           -   rf_wsel: Select one of 8 MXCU local registers to write to. Note that some registers have special jobs. See vwr2a_ISA doc.
           -   rf_we: Enable writing to local registers
           -   alu_op: Perform one of the ALU operations listed in the MXCU_ALU_OPS enum
           -   muxb_sel: Select input B to ALU (see MXCU_MUXB_SEL enum for options)
           -   muxa_sel: Select input A to ALU (see MXCU_MUXA_SEL enum for options)
        
        '''
        binary_vwr_row_we = ""
        for b in vwr_row_we:
            binary_vwr_row_we += (np.binary_repr(b))
        self.vwr_row_we = binary_vwr_row_we
        self.vwr_sel = np.binary_repr(vwr_sel,2)
        self.srf_sel = np.binary_repr(srf_sel,3)
        self.alu_srf_write = np.binary_repr(alu_srf_write,2)
        self.srf_we = np.binary_repr(srf_we, 1)
        self.rf_wsel = np.binary_repr(rf_wsel, width=3)
        self.rf_we = np.binary_repr(rf_we,width=1)
        self.alu_op = np.binary_repr(alu_op,3)
        self.muxb_sel = np.binary_repr(muxb_sel,4)
        self.muxa_sel = np.binary_repr(muxa_sel,4)
        self.word = "".join((self.muxa_sel, self.muxb_sel, self.alu_op, self.rf_we, self.rf_wsel, self.srf_we, self.alu_srf_write, self.srf_sel, self.vwr_sel, self.vwr_row_we))
    
    def get_word(self):
        return self.word
    
    def set_word(self, word):
        '''Set the binary configuration word of the kernel memory'''
        self.word = word
        self.vwr_row_we = word[23:]
        self.vwr_sel = word[21:23]
        self.srf_sel = word[18:21]
        self.alu_srf_write = word[16:18]
        self.srf_we = word[15:16]
        self.rf_wsel = word[12:15]
        self.rf_we = word[11:12]
        self.alu_op = word[8:11]
        self.muxb_sel = word[4:8]
        self.muxa_sel = word[0:4]
        
    
    def decode_word(self):
        '''Get the configuration word parameters from the binary word'''

        vwr_sel = int(self.vwr_sel,2)
        srf_sel = int(self.srf_sel,2)
        alu_srf_write = int(self.alu_srf_write,2)
        srf_we = int(self.srf_we,2)
        rf_wsel = int(self.rf_wsel,2)
        rf_we = int(self.rf_we,2)
        alu_op = int(self.alu_op,2)
        muxb_sel = int(self.muxb_sel,2)
        muxa_sel = int(self.muxa_sel,2)
        
        # Return one-hot veectors as numpy arrays of integers or booleans
        one_hot_vwr_row_we = []
        for i in range(len(self.vwr_row_we)):
            one_hot_vwr_row_we.append(int(self.vwr_row_we[i:i+1],2))
        
        return one_hot_vwr_row_we, vwr_sel, srf_sel, alu_srf_write, srf_we, rf_wsel, rf_we, alu_op, muxb_sel, muxa_sel