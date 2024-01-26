"""ker_conf.py: Data structures and objects emulating the kernel configuration of the VWR2A architecture"""
__author__      = "Lara Orlandic"
__email__       = "lara.orlandic@epfl.ch"

import numpy as np

# Configuration register (CREG) / instruction memory sizes of specialized slots
KER_CONF_N_REG = 16

# Widths of instructions of each specialized slot in bits
KER_CONF_IMEM_WIDTH = 21

# KERNEL CONFIGURATION #
class KER_CONF:
    '''Kernel memory: Keeps track of which kernels are loaded into the IMEM of VWR2A'''
    def __init__(self):
        self.IMEM = np.zeros(KER_CONF_N_REG,dtype="S{0}".format(KER_CONF_IMEM_WIDTH))
        # Initialize kernel memory with zeros
        for i, instruction in enumerate(self.IMEM):
            self.IMEM[i] = np.binary_repr(0,width=KER_CONF_IMEM_WIDTH)
    
    def set_word(self, kmem_word, pos):
        '''Set the IMEM index at integer pos to the binary kmem word'''
        assert (pos>0), "Kernel word 0 is reserved; need to pick a position >0 and <16"
        
        self.IMEM[pos] = np.binary_repr(kmem_word,width=KER_CONF_IMEM_WIDTH)
    
    def set_params(self, num_instructions=0, imem_add_start=0, column_usage=0, srf_spm_addres=0, pos=1):
        '''Set the IMEM index at integer pos to the configuration parameters.
        See KMEM_WORD initializer for implementation details.
        '''
        
        assert (pos>0), "Kernel word 0 is reserved; need to pick a position >0 and <16"
        assert (num_instructions>0) & (num_instructions<64), "Invalid kernel; number of instructions is either negative or too big"
        assert (column_usage>0), "The column attribute must be one-hot encoded"
        
        # Note: The number of instructions encoded in the kmem word is always one less than the actual number of instructions
        n_instr_kmem = num_instructions-1
        
        kmem_word = KMEM_WORD(n_instr_kmem, imem_add_start, column_usage, srf_spm_addres)
        self.IMEM[pos] = kmem_word.get_word()
    
    def get_params(self, pos):
        '''Get the kernel parameters at position pos in the kernel memory'''
        kmem_word = KMEM_WORD()
        kmem_word.set_word(self.IMEM[pos])
        n_instr, imem_add, col, spm_add = kmem_word.decode_word()
        return n_instr, imem_add, col, spm_add
    
    def get_kernel_info(self, pos):
        '''Get the kernel implementation details at position pos in the kernel memory'''
        kmem_word = KMEM_WORD()
        kmem_word.set_word(self.IMEM[pos])
        n_instr, imem_add, col, spm_add = kmem_word.decode_word()
        
        # Note: The number of instructions encoded in the kmem word is always one less than the actual number of instructions
        n_instr += 1
        
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