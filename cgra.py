import numpy as np
from enum import Enum

from ctypes import c_int32
import csv


from kernels import *

# CGRA top-level parameters
N_ROW      = 4
N_COL      = 2
N_VWR_PER_COL = 3

# Local data register (DREG) sizes of specialized slots
RC_NUM_DREG = 2
LCU_NUM_DREG = 4

# Configuration register (CREG) / instruction memory sizes of specialized slots
RC_NUM_CREG = 64
LSU_NUM_CREG = 64
LCU_NUM_CREG = 64
MXCU_NUM_CREG = 64
KER_CONF_N_REG = 16
IMEM_N_LINES = 512

# Widths of instruction memories of each specialized slot in bits
KER_CONF_IMEM_WIDTH = 21
LCU_IMEM_WIDTH = 20
LSU_IMEM_WIDTH = 20
RC_IMEM_WIDTH = 18
MXCU_IMEM_WIDTH = 27

# Scratchpad memory configuration
SP_NWORDS = 128
SP_NLINES = 64

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


#### SPECIALIZED SLOTS: Sub-modules of the VWR2A top module that each perform their own purpos and have their own ISA ######

# KERNEL CONFIGURATION #
class KER_CONF:
    '''Kernel memory: Keeps track of which kernels are loaded into the IMEM of VWR2A'''
    def __init__(self):
        self.IMEM = np.zeros(KER_CONF_N_REG,dtype="S{0}".format(KER_CONF_IMEM_WIDTH))
        # Initialize kernel memory with zeros
        for i, instruction in enumerate(self.IMEM):
            self.IMEM[i] = np.binary_repr(0,width=KER_CONF_IMEM_WIDTH)
    
    def set_kernel_word(self, kmem_word, pos):
        '''Set the IMEM index at integer pos to the binary kmem word'''
        self.IMEM[pos] = np.binary_repr(kmem_word,width=KER_CONF_IMEM_WIDTH)
    
    def set_kernel_params(self, num_instructions, imem_add_start, column_usage, srf_spm_addres, pos):
        '''Set the IMEM index at integer pos to the configuration parameters.
        See KMEM_WORD initializer for implementation details.
        '''
        kmem_word = KMEM_WORD(num_instructions, imem_add_start, column_usage, srf_spm_addres)
        self.IMEM[pos] = kmem_word.get_word()
    
    def get_kernel_info(self, pos):
        '''Get the kernel implementation details at position pos in the kernel memory'''
        kmem_word = KMEM_WORD()
        kmem_word.set_word(self.IMEM[pos])
        n_instr, imem_add, col, spm_add = kmem_word.decode_word()
        if col == 1:
            col_disp = 0
        elif col == 2:
            col_disp = 1
        elif col == 3:
            col_disp = "both"
        print("This kernel uses {0} instruction words starting at IMEM address {1}.\nIt uses column(s): {2}.\nThe SRF is located in SPM bank {3}.".format(n_instr, imem_add, col_disp, spm_add))
        
    def get_word_in_hex(self, pos):
        '''Get the hexadecimal representation of the word at index pos in the kernel config IMEM'''
        return(hex(int(self.IMEM[pos],2)))
        
    
        
class KMEM_WORD:
    def __init__(self, num_instructions=0, imem_add_start=0, column_usage=0, srf_spm_addres=0):
        '''Generate a binary kmem instruction word from its configuration paramerers:
        
           -   num_instructions: number of IMEM lines the kernel occupies (0 to 63)
           -   imem_add_start: start address of the kernel in IMEM (0 to 511)
           -   column_usage: integrer representing one-hot column usage of the kernel:
               -    1 for column 0
               -    2 for column 1
               -    3 for both columns
           -   srf_spm_address: address of SPM that SRF occupies (0 to 15)
        
        '''
        self.num_instructions = np.binary_repr(num_instructions, width=6)
        self.imem_add_start = np.binary_repr(imem_add_start, width=9)
        self.column_usage = np.binary_repr(column_usage,width=2)
        self.srf_spm_addres = np.binary_repr(srf_spm_addres,4)
        self.word = "".join((self.srf_spm_addres,self.column_usage,self.imem_add_start,self.num_instructions))
    
    def get_word(self):
        return self.word
    
    def set_word(self, word):
        '''Set the binary configuration word of the kernel memory'''
        self.word = word
        self.num_instructions = word[15:]
        self.imem_add_start = word[6:15]
        self.column_usage = word[4:6]
        self.srf_spm_addres = word[0:4]
        
    
    def decode_word(self):
        '''Get the configuration word parameters from the binary word'''
        n_instr = int(self.num_instructions, 2)
        imem_add = int(self.imem_add_start ,2)
        col = int(self.column_usage, 2)
        spm_add = int(self.srf_spm_addres, 2)
        
        return n_instr, imem_add, col, spm_add

# LOOP CONTROL UNIT (LCU) #

class LCU_IMEM:
    '''Instruction memory of the Loop Control Unit'''
    def __init__(self):
        self.IMEM = np.zeros(LCU_NUM_CREG,dtype="S{0}".format(LCU_IMEM_WIDTH))
        # Initialize kernel memory with zeros
        for i, instruction in enumerate(self.IMEM):
            self.IMEM[i] = np.binary_repr(0,width=LCU_IMEM_WIDTH)
    
    def set_kernel_word(self, kmem_word, pos):
        '''Set the IMEM index at integer pos to the binary kmem word'''
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
    
# LOAD-STORE UNIT (LSU) #


class CGRA:
    def __init__( self, kernel, memory, inputs, outputs ):
        self.cells      = [[ PE( self, c,r) for r in range(N_ROWS)] for c in range(N_COLS)]
        self.instrs     = ker_parse( kernel )
        self.memory     = memory
        self.inputs     = inputs
        self.outputs    = outputs
        self.instr2exec = 0
        self.cycles     = 0
        self.load_idx   = [0]*N_COLS
        self.store_idx  = [0]*N_COLS
        self.exit       = False

    def run( self, pr, limit ):
        steps = 0
        while not self.step(pr):
            print("-------")
            steps += 1
            if steps > limit:
                print("EXECUTION LIMIT REACHED (",limit,"steps)")
                print("Extend the execution by calling the run with argument limit=<steps>.")
                break
        return self.outputs, self.memory

    def step( self, prs="ROUT" ):
        for r in range(N_ROWS):
            for c in range(N_COLS):
                self.cells[r][c].update()
        if PRINT_OUTS: print("Instr = ", self.cycles, "(",self.instr2exec,")")
        for r in range(N_ROWS):
            for c in range(N_COLS):
                op =  self.instrs[self.instr2exec].ops[r][c]
                b ,e = self.cells[r][c].exec( op )
                if b != 0: self.instr2exec = b - 1 #To avoid more logic afterwards
                if e != 0: self.exit = True
            outs    = [ self.cells[r][i].out        for i in range(N_COLS) ]
            insts   = [ self.cells[r][i].instr      for i in range(N_COLS) ]
            ops     = [ self.cells[r][i].op         for i in range(N_COLS) ]
            reg     = [[ self.cells[r][i].regs[regs[x]]   for i in range(N_COLS) ] for x in range(len(regs)) ]
            print_out( prs, outs, insts, ops, reg )

        self.instr2exec += 1
        self.cycles += 1
        return self.exit

    def get_neighbour_address( self, r, c, dir ):
        n_r = r
        n_c = c
        if dir == "RCL": n_c = c - 1 if c > 0 else MAX_COL
        if dir == "RCR": n_c = c + 1 if c < MAX_COL else 0
        if dir == "RCT": n_r = r - 1 if r > 0 else MAX_ROW
        if dir == "RCB": n_r = r + 1 if r < MAX_ROW else 0
        return n_r, n_c

    def get_neighbour_out( self, r, c, dir ):
        n_r, n_c = self.get_neighbour_address( r, c, dir )
        return self.cells[n_r][n_c].get_out()

    def get_neighbour_flag( self, r, c, dir, flag ):
        n_r, n_c = self.get_neighbour_address( r, c, dir )
        return self.cells[n_r][n_c].get_flag( flag )

    def load_direct( self, c ):
        ret = self.inputs[  self.load_idx[c]][ c ]
        self.load_idx[c] += 1
        return int(ret)

    def store_direct( self, c, val ):
        if self.store_idx[c] >= len(self.outputs): self.outputs.append([0]*N_COLS)
        self.outputs[ self.store_idx[c] ][c] = val
        self.store_idx[c] += 1

    def load_indirect( self, add ):
        for row in self.memory[1:]:
            if int(row[0]) == add:
                return int(row[1])
        return -1

    def store_indirect( self, add, val):
        for i in range(1,len(self.memory)):
            if int(self.memory[i][0]) == add:
                self.memory[i][1] = val
                return
        self.memory.append([add, val])
        return

class PE:
    def __init__( self, parent, row, col ):
        self.parent = parent
        self.row = row
        self.col = col
        self.flags      = { "sign"   : 0,
                            "zero"   : 0,
                            "branch" : 0,
                            "exit"   : 0}
        self.instr      = ""
        self.old_out    = 0
        self.out        = 0
        self.regs       = {'R0':0, 'R1':0, 'R2':0, 'R3':0 }
        self.op         = ""
        self.instr      = ""

    def get_out( self ):
        return self.old_out

    def get_flag( self, flag ):
        return self.flags[flag]

    def fetch_val( self, val):
        if val.lstrip('-+').isnumeric():
            return int(val)
        if val == 'ROUT':
            return int( self.old_out)
        if val == 'ZERO':
            return 0
        if val in self.regs:
            return int( self.regs[val])
        return int(self.parent.get_neighbour_out( self.row, self.col, val ))

    def fetch_flag( self, dir, flag ):
        if dir == 'ROUT':
            return int( self.old_out)
        return int(self.parent.get_neighbour_flag( self.row, self.col, dir, flag ))

    def exec( self,  instr ):
        self.run_instr(instr)
        return self.flags["branch"], self.flags["exit"]

    def update( self):
        self.old_out = self.out
        self.flags["zero"]      = 1 if self.out == 0 else 0
        self.flags["sign"]      = 1 if self.out <  0 else 0
        self.flags["branch"]    = 0

    def run_instr( self, instr):
        instr   = instr.replace(',', ' ')   # Remove the commas so we can speparate arguments by spaces
        self.instr = instr                  # Save this string as instruction to show
        instr   = instr.split()             # Split into chunks
        try:
            self.op      = instr[0]
        except:
            self.op = instr

        if self.op in self.ops_arith:
            des     = instr[1]
            val1    = self.fetch_val( instr[2] )
            val2    = self.fetch_val( instr[3] )
            ret     = self.ops_arith[self.op]( val1, val2)
            if des in self.regs: self.regs[des] = ret
            self.out = ret

        elif self.op in self.ops_cond:
            des     = instr[1]
            val1    = self.fetch_val( instr[2] )
            val2    = self.fetch_val( instr[3] )
            src     = instr[4]
            method  = self.ops_cond[self.op]
            ret     = method(self, val1, val2, src)
            if des in self.regs: self.regs[des] = ret
            self.out = ret

        elif self.op in self.ops_branch:
            val1    = self.fetch_val( instr[1] )
            val2    = self.fetch_val( instr[2] )
            branch  = self.fetch_val( instr[3] )
            method = self.ops_branch[self.op]
            method(self, val1, val2, branch)

        elif self.op in self.ops_lwd:
            des = instr[1]
            ret = self.parent.load_direct( self.col )
            if des in self.regs: self.regs[des] = ret
            self.out = ret

        elif self.op in self.ops_swd:
            val = self.fetch_val( instr[1] )
            self.parent.store_direct( self.col, val )

        elif self.op in self.ops_lwi:
            des = instr[1]
            add = self.fetch_val( instr[2] )
            ret = self.parent.load_indirect(add)
            if des in self.regs: self.regs[des] = ret
            self.out = ret

        elif self.op in self.ops_swi:
            add = self.fetch_val( instr[1] )
            val = self.fetch_val( instr[2] )
            self.parent.store_indirect( add, val )
            pass

        elif self.op in self.ops_nop:
            pass # Intentional

        elif self.op in self.ops_jump:
            self.flags['branch'] = val1

        elif self.op in self.ops_exit:
            self.flags['exit'] = 1

    def sadd( val1, val2 ):
        return c_int32( val1 + val2 ).value

    def ssub( val1, val2 ):
        return c_int32( val1 - val2 ).value

    def smul( val1, val2 ):
        return c_int32( val1 * val2 ).value

    def fxpmul( val1, val2 ):
        print("Better luck next time")
        return 0

    def slt( val1, val2 ):
        return c_int32(val1 << val2).value

    def srt( val1, val2 ):
        interm_result = (c_int32(val1).value & MAX_32b)
        return c_int32(interm_result >> val2).value

    def sra( val1, val2 ):
        return c_int32(val1 >> val2).value

    def lor( val1, val2 ):
        return c_int32( val1 | val2).value

    def land( val1, val2 ):
        return c_int32( val1 & val2).value

    def lxor( val1, val2 ):
        return c_int32( ((val1& MAX_32b) ^ (val2& MAX_32b)) & MAX_32b).value

    def lnand( val1, val2 ):
        return c_int32( ~( val1 & val2 ) & MAX_32b ).value

    def lnor( val1, val2 ):
         return c_int32( ~( val1 | val2 ) & MAX_32b ).value

    def lxnor( val1, val2 ):
        return c_int32( ~( val1 ^ val2 ) & MAX_32b ).value

    def bsfa( self, val1, val2, src):
        flag = self.fetch_flag( src, 'sign')
        return val1 if flag == 1 else val2

    def bzfa( self,  val1, val2, src):
        flag = self.fetch_flag( src, 'zero')
        return val1 if flag == 1 else val2

    def beq( self,  val1, val2, branch ):
        self.flags['branch'] = branch if val1 == val2 else self.flags['branch']

    def bne( self,  val1, val2, branch ):
        self.flags['branch'] = branch if val1 != val2 else self.flags['branch']

    def bge( self,  val1, val2, branch ):
        self.flags['branch'] = branch if val1 >= val2 else self.flags['branch']
    
    def blt( self,  val1, val2, branch ):
        self.flags['branch'] = branch if val1 < val2 else self.flags['branch']

    ops_arith   = { 'SADD'      : sadd,
                    'SSUB'      : ssub,
                    'SMUL'      : smul,
                    'FXPMUL'    : fxpmul,
                    'SLT'       : slt,
                    'SRT'       : srt,
                    'SRA'       : sra,
                    'LOR'       : lor,
                    'LAND'      : land,
                    'LXOR'      : lxor,
                    'LNAND'     : lnand,
                    'LNOR'      : lnor,
                    'LXNOR'     : lxnor }

    ops_cond    = { 'BSFA'      : bsfa,
                    'BZFA'      : bzfa }

    ops_branch  = { 'BEQ'       : beq,
                    'BNE'       : bne,
                    'BLT'       : blt,
                    'BGE'       : bge }

    ops_lwd     = { 'LWD'       : '' }
    ops_swd     = { 'SWD'       : '' }
    ops_lwi     = { 'LWI'       : '' }
    ops_swi     = { 'SWI'       : '' }

    ops_nop     = { 'NOP'       : '' }
    ops_jump    = { 'JUMP'      : '' }
    ops_exit    = { 'EXIT'      : '' }

def run( kernel, version="", pr="ROUT", limit=100 ):
    ker = []
    inp = []
    oup = []
    mem = []

    with open( kernel + "/"+FILENAME_INSTR+version+EXT, 'r') as f:
        for row in csv.reader(f): ker.append(row)
    with open( kernel + "/"+FILENAME_INP+EXT, 'r') as f:
        for row in csv.reader(f): inp.append(row)
    with open( kernel + "/"+FILENAME_MEM+EXT, 'r') as f:
        for row in csv.reader(f): mem.append(row)

    oup, mem = CGRA( ker, mem, inp, oup ).run(pr, limit)

    with open( kernel + "/"+FILENAME_MEM_O+EXT, 'w+') as f:
        for row in mem: csv.writer(f).writerow(row)
    with open( kernel + "/"+FILENAME_OUP+EXT, 'w+') as f:
        for row in oup: csv.writer(f).writerow(row)

    print("\n\nEND")

