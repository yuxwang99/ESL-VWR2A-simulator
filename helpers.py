"""helpers.py: Helper functions for dealing with VWR2A simulator file I/O, etc."""
__author__      = "Lara Orlandic"
__email__       = "lara.orlandic@epfl.ch"

from src import *

def dataframe_to_header_file(df,kernel_path):
    '''Take a csv of ISA instructons (developed in kernel mapping) and make a header file to be loaded into the RTL testbench'''
    
    with open(kernel_path + 'dsip_bitstream.h', 'w+') as file:
        file.write("#ifndef _DSIP_BITSTREAM_H_\n#define _DSIP_BITSTREAM_H_\n\n#include <stdint.h>\n\n#include \"dsip.h\"\n\n")

        # Write kmem bitstream
        file.write("uint32_t dsip_kmem_bitstream[DSIP_KMEM_SIZE] = {\n")
        for i in range(KER_CONF_N_REG):
            if i<KER_CONF_N_REG-1:
                file.write("  {0},\n".format(df.KMEM.iloc[i]))
            else:
                file.write("  {0}\n".format(df.KMEM.iloc[i]))
        file.write("};\n\n\n")

        # Write LCU bitstream
        file.write("uint32_t dsip_lcu_imem_bitstream[DSIP_IMEM_SIZE] = {\n")
        for i in range(IMEM_N_LINES):
            if i<IMEM_N_LINES-1:
                file.write("  {0},\n".format(df.LCU.iloc[i]))
            else:
                file.write("  {0}\n".format(df.LCU.iloc[i]))
        file.write("};\n\n\n")

        # Write LSU bitstream
        file.write("uint32_t dsip_lsu_imem_bitstream[DSIP_IMEM_SIZE] = {\n")
        for i in range(IMEM_N_LINES):
            if i<IMEM_N_LINES-1:
                file.write("  {0},\n".format(df.LSU.iloc[i]))
            else:
                file.write("  {0}\n".format(df.LSU.iloc[i]))
        file.write("};\n\n\n")

        # Write MXCU bitstream
        file.write("uint32_t dsip_mxcu_imem_bitstream[DSIP_IMEM_SIZE] = {\n")
        for i in range(IMEM_N_LINES):
            if i<IMEM_N_LINES-1:
                file.write("  {0},\n".format(df.MXCU.iloc[i]))
            else:
                file.write("  {0}\n".format(df.MXCU.iloc[i]))
        file.write("};\n\n\n")

        # Write bitstream of all RCs concatenated
        file.write("uint32_t dsip_rcs_imem_bitstream[4*DSIP_IMEM_SIZE] = {\n")
        #RC0
        for i in range(IMEM_N_LINES):
            file.write("  {0},\n".format(df.RC0.iloc[i]))
        #RC1
        for i in range(IMEM_N_LINES):
            file.write("  {0},\n".format(df.RC1.iloc[i]))
        #RC2
        for i in range(IMEM_N_LINES):
            file.write("  {0},\n".format(df.RC2.iloc[i]))
        #RC3
        for i in range(IMEM_N_LINES):
            if i<IMEM_N_LINES-1:
                file.write("  {0},\n".format(df.RC3.iloc[i]))
            else:
                file.write("  {0}\n".format(df.RC3.iloc[i]))
        file.write("};\n\n\n")
        file.write("#endif // _DSIP_BITSTREAM_H_")