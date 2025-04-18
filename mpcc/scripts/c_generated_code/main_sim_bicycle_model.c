/*
 * Copyright (c) The acados authors.
 *
 * This file is part of acados.
 *
 * The 2-Clause BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.;
 */


// standard
#include <stdio.h>
#include <stdlib.h>
// acados
#include "acados/utils/print.h"
#include "acados/utils/math.h"
#include "acados_c/sim_interface.h"
#include "acados_sim_solver_bicycle_model.h"

#define NX     BICYCLE_MODEL_NX
#define NZ     BICYCLE_MODEL_NZ
#define NU     BICYCLE_MODEL_NU
#define NP     BICYCLE_MODEL_NP


int main()
{
    int status = 0;
    bicycle_model_sim_solver_capsule *capsule = bicycle_model_acados_sim_solver_create_capsule();
    status = bicycle_model_acados_sim_create(capsule);

    if (status)
    {
        printf("acados_create() returned status %d. Exiting.\n", status);
        exit(1);
    }

    sim_config *acados_sim_config = bicycle_model_acados_get_sim_config(capsule);
    sim_in *acados_sim_in = bicycle_model_acados_get_sim_in(capsule);
    sim_out *acados_sim_out = bicycle_model_acados_get_sim_out(capsule);
    void *acados_sim_dims = bicycle_model_acados_get_sim_dims(capsule);

    // initial condition
    double x_current[NX];
    x_current[0] = 0.0;
    x_current[1] = 0.0;
    x_current[2] = 0.0;
    x_current[3] = 0.0;
    x_current[4] = 0.0;
    x_current[5] = 0.0;

  
    x_current[0] = 0;
    x_current[1] = 0;
    x_current[2] = 0;
    x_current[3] = 0;
    x_current[4] = 0;
    x_current[5] = 0;
    
  


    // initial value for control input
    double u0[NU];
    u0[0] = 0.0;
    u0[1] = 0.0;
    u0[2] = 0.0;
    // set parameters
    double p[NP];
    p[0] = 0;
    p[1] = 0;
    p[2] = 0;
    p[3] = 0;
    p[4] = 0;
    p[5] = 0;
    p[6] = 0;
    p[7] = 0;
    p[8] = 0;
    p[9] = 0;
    p[10] = 0;
    p[11] = 0;
    p[12] = 0;
    p[13] = 0;
    p[14] = 0;
    p[15] = 0;
    p[16] = 0;
    p[17] = 0;
    p[18] = 0;
    p[19] = 0;
    p[20] = 0;
    p[21] = 0;
    p[22] = 0;
    p[23] = 0;
    p[24] = 0;
    p[25] = 0;
    p[26] = 0;
    p[27] = 0;
    p[28] = 0;
    p[29] = 0;
    p[30] = 0;
    p[31] = 0;
    p[32] = 0;
    p[33] = 0;
    p[34] = 0;
    p[35] = 0;
    p[36] = 0;
    p[37] = 0;
    p[38] = 0;
    p[39] = 0;
    p[40] = 0;
    p[41] = 0;
    p[42] = 0;
    p[43] = 0;
    p[44] = 0;
    p[45] = 0;
    p[46] = 0;
    p[47] = 0;
    p[48] = 0;
    p[49] = 0;
    p[50] = 0;
    p[51] = 0;
    p[52] = 0;
    p[53] = 0;
    p[54] = 0;
    p[55] = 0;
    p[56] = 0;
    p[57] = 0;
    p[58] = 0;
    p[59] = 0;
    p[60] = 0;
    p[61] = 0;
    p[62] = 0;
    p[63] = 0;
    p[64] = 0;
    p[65] = 0;
    p[66] = 0;
    p[67] = 0;
    p[68] = 0;
    p[69] = 0;
    p[70] = 0;
    p[71] = 0;
    p[72] = 0;
    p[73] = 0;
    p[74] = 0;
    p[75] = 0;
    p[76] = 0;
    p[77] = 0;
    p[78] = 0;
    p[79] = 0;
    p[80] = 0;
    p[81] = 0;
    p[82] = 0;
    p[83] = 0;
    p[84] = 0;
    p[85] = 0;
    p[86] = 0;
    p[87] = 0;
    p[88] = 0;
    p[89] = 0;
    p[90] = 0;
    p[91] = 0;
    p[92] = 0;
    p[93] = 0;
    p[94] = 0;
    p[95] = 0;
    p[96] = 0;
    p[97] = 0;
    p[98] = 0;
    p[99] = 0;
    p[100] = 0;
    p[101] = 0;
    p[102] = 0;
    p[103] = 0;
    p[104] = 0;
    p[105] = 0;
    p[106] = 0;
    p[107] = 0;
    p[108] = 0;
    p[109] = 0;
    p[110] = 0;
    p[111] = 0;
    p[112] = 0;
    p[113] = 0;
    p[114] = 0;
    p[115] = 0;
    p[116] = 0;
    p[117] = 0;
    p[118] = 0;
    p[119] = 0;
    p[120] = 0;
    p[121] = 0;
    p[122] = 0;
    p[123] = 0;
    p[124] = 0;
    p[125] = 0;
    p[126] = 0;
    p[127] = 0;
    p[128] = 0;
    p[129] = 0;
    p[130] = 0;
    p[131] = 0;
    p[132] = 0;
    p[133] = 0;
    p[134] = 0;
    p[135] = 0;
    p[136] = 0;
    p[137] = 0;
    p[138] = 0;
    p[139] = 0;
    p[140] = 0;
    p[141] = 0;
    p[142] = 0;
    p[143] = 0;
    p[144] = 0;
    p[145] = 0;
    p[146] = 0;
    p[147] = 0;
    p[148] = 0;
    p[149] = 0;
    p[150] = 0;
    p[151] = 0;
    p[152] = 0;
    p[153] = 0;
    p[154] = 0;
    p[155] = 0;
    p[156] = 0;
    p[157] = 0;
    p[158] = 0;
    p[159] = 0;
    p[160] = 0;
    p[161] = 0;
    p[162] = 0;
    p[163] = 0;
    p[164] = 0;
    p[165] = 0;
    p[166] = 0;
    p[167] = 0;
    p[168] = 0;
    p[169] = 0;
    p[170] = 0;
    p[171] = 0;
    p[172] = 0;
    p[173] = 0;
    p[174] = 0;
    p[175] = 0;
    p[176] = 0;
    p[177] = 0;
    p[178] = 0;
    p[179] = 0;
    p[180] = 0;
    p[181] = 0;
    p[182] = 0;
    p[183] = 0;
    p[184] = 0;
    p[185] = 0;
    p[186] = 0;
    p[187] = 0;
    p[188] = 0;
    p[189] = 0;
    p[190] = 0;
    p[191] = 0;
    p[192] = 0;
    p[193] = 0;
    p[194] = 0;
    p[195] = 0;
    p[196] = 0;
    p[197] = 0;
    p[198] = 0;
    p[199] = 0;
    p[200] = 0;
    p[201] = 0;
    p[202] = 0;
    p[203] = 0;
    p[204] = 0;
    p[205] = 0;
    p[206] = 0;
    p[207] = 0;
    p[208] = 0;
    p[209] = 0;
    p[210] = 0;
    p[211] = 0;
    p[212] = 0;
    p[213] = 0;
    p[214] = 0;
    p[215] = 0;
    p[216] = 0;
    p[217] = 0;

    bicycle_model_acados_sim_update_params(capsule, p, NP);
  

  


    int n_sim_steps = 3;
    // solve ocp in loop
    for (int ii = 0; ii < n_sim_steps; ii++)
    {
        // set inputs
        sim_in_set(acados_sim_config, acados_sim_dims,
            acados_sim_in, "x", x_current);
        sim_in_set(acados_sim_config, acados_sim_dims,
            acados_sim_in, "u", u0);

        // solve
        status = bicycle_model_acados_sim_solve(capsule);
        if (status != ACADOS_SUCCESS)
        {
            printf("acados_solve() failed with status %d.\n", status);
        }

        // get outputs
        sim_out_get(acados_sim_config, acados_sim_dims,
               acados_sim_out, "x", x_current);

    

        // print solution
        printf("\nx_current, %d\n", ii);
        for (int jj = 0; jj < NX; jj++)
        {
            printf("%e\n", x_current[jj]);
        }
    }

    printf("\nPerformed %d simulation steps with acados integrator successfully.\n\n", n_sim_steps);

    // free solver
    status = bicycle_model_acados_sim_free(capsule);
    if (status) {
        printf("bicycle_model_acados_sim_free() returned status %d. \n", status);
    }

    bicycle_model_acados_sim_solver_free_capsule(capsule);

    return status;
}
