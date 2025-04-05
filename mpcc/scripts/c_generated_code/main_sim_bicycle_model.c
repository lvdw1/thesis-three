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

  
    x_current[0] = -71.69;
    x_current[1] = -20.11;
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
    p[218] = 0;
    p[219] = 0;
    p[220] = 0;
    p[221] = 0;
    p[222] = 0;
    p[223] = 0;
    p[224] = 0;
    p[225] = 0;
    p[226] = 0;
    p[227] = 0;
    p[228] = 0;
    p[229] = 0;
    p[230] = 0;
    p[231] = 0;
    p[232] = 0;
    p[233] = 0;
    p[234] = 0;
    p[235] = 0;
    p[236] = 0;
    p[237] = 0;
    p[238] = 0;
    p[239] = 0;
    p[240] = 0;
    p[241] = 0;
    p[242] = 0;
    p[243] = 0;
    p[244] = 0;
    p[245] = 0;
    p[246] = 0;
    p[247] = 0;
    p[248] = 0;
    p[249] = 0;
    p[250] = 0;
    p[251] = 0;
    p[252] = 0;
    p[253] = 0;
    p[254] = 0;
    p[255] = 0;
    p[256] = 0;
    p[257] = 0;
    p[258] = 0;
    p[259] = 0;
    p[260] = 0;
    p[261] = 0;
    p[262] = 0;
    p[263] = 0;
    p[264] = 0;
    p[265] = 0;
    p[266] = 0;
    p[267] = 0;
    p[268] = 0;
    p[269] = 0;
    p[270] = 0;
    p[271] = 0;
    p[272] = 0;
    p[273] = 0;
    p[274] = 0;
    p[275] = 0;
    p[276] = 0;
    p[277] = 0;
    p[278] = 0;
    p[279] = 0;
    p[280] = 0;
    p[281] = 0;
    p[282] = 0;
    p[283] = 0;
    p[284] = 0;
    p[285] = 0;
    p[286] = 0;
    p[287] = 0;
    p[288] = 0;
    p[289] = 0;
    p[290] = 0;
    p[291] = 0;
    p[292] = 0;
    p[293] = 0;
    p[294] = 0;
    p[295] = 0;
    p[296] = 0;
    p[297] = 0;
    p[298] = 0;
    p[299] = 0;
    p[300] = 0;
    p[301] = 0;
    p[302] = 0;
    p[303] = 0;
    p[304] = 0;
    p[305] = 0;
    p[306] = 0;
    p[307] = 0;
    p[308] = 0;
    p[309] = 0;
    p[310] = 0;
    p[311] = 0;
    p[312] = 0;
    p[313] = 0;
    p[314] = 0;
    p[315] = 0;
    p[316] = 0;
    p[317] = 0;
    p[318] = 0;
    p[319] = 0;
    p[320] = 0;
    p[321] = 0;
    p[322] = 0;
    p[323] = 0;
    p[324] = 0;
    p[325] = 0;
    p[326] = 0;
    p[327] = 0;
    p[328] = 0;
    p[329] = 0;
    p[330] = 0;
    p[331] = 0;
    p[332] = 0;
    p[333] = 0;
    p[334] = 0;
    p[335] = 0;
    p[336] = 0;
    p[337] = 0;
    p[338] = 0;
    p[339] = 0;
    p[340] = 0;
    p[341] = 0;
    p[342] = 0;
    p[343] = 0;
    p[344] = 0;
    p[345] = 0;
    p[346] = 0;
    p[347] = 0;
    p[348] = 0;
    p[349] = 0;
    p[350] = 0;
    p[351] = 0;
    p[352] = 0;
    p[353] = 0;
    p[354] = 0;
    p[355] = 0;
    p[356] = 0;
    p[357] = 0;
    p[358] = 0;
    p[359] = 0;
    p[360] = 0;
    p[361] = 0;
    p[362] = 0;
    p[363] = 0;
    p[364] = 0;
    p[365] = 0;
    p[366] = 0;
    p[367] = 0;
    p[368] = 0;
    p[369] = 0;
    p[370] = 0;
    p[371] = 0;
    p[372] = 0;
    p[373] = 0;
    p[374] = 0;
    p[375] = 0;
    p[376] = 0;
    p[377] = 0;
    p[378] = 0;
    p[379] = 0;
    p[380] = 0;
    p[381] = 0;
    p[382] = 0;
    p[383] = 0;
    p[384] = 0;
    p[385] = 0;
    p[386] = 0;
    p[387] = 0;
    p[388] = 0;
    p[389] = 0;
    p[390] = 0;
    p[391] = 0;
    p[392] = 0;
    p[393] = 0;
    p[394] = 0;
    p[395] = 0;
    p[396] = 0;
    p[397] = 0;
    p[398] = 0;
    p[399] = 0;
    p[400] = 0;
    p[401] = 0;
    p[402] = 0;
    p[403] = 0;
    p[404] = 0;
    p[405] = 0;
    p[406] = 0;
    p[407] = 0;
    p[408] = 0;
    p[409] = 0;
    p[410] = 0;
    p[411] = 0;
    p[412] = 0;
    p[413] = 0;
    p[414] = 0;
    p[415] = 0;
    p[416] = 0;
    p[417] = 0;
    p[418] = 0;
    p[419] = 0;
    p[420] = 0;
    p[421] = 0;
    p[422] = 0;
    p[423] = 0;
    p[424] = 0;
    p[425] = 0;
    p[426] = 0;
    p[427] = 0;
    p[428] = 0;
    p[429] = 0;
    p[430] = 0;
    p[431] = 0;
    p[432] = 0;
    p[433] = 0;
    p[434] = 0;
    p[435] = 0;
    p[436] = 0;
    p[437] = 0;
    p[438] = 0;
    p[439] = 0;
    p[440] = 0;
    p[441] = 0;
    p[442] = 0;
    p[443] = 0;
    p[444] = 0;
    p[445] = 0;
    p[446] = 0;
    p[447] = 0;
    p[448] = 0;
    p[449] = 0;
    p[450] = 0;
    p[451] = 0;
    p[452] = 0;
    p[453] = 0;
    p[454] = 0;
    p[455] = 0;
    p[456] = 0;
    p[457] = 0;
    p[458] = 0;
    p[459] = 0;
    p[460] = 0;
    p[461] = 0;
    p[462] = 0;
    p[463] = 0;
    p[464] = 0;
    p[465] = 0;
    p[466] = 0;
    p[467] = 0;
    p[468] = 0;
    p[469] = 0;
    p[470] = 0;
    p[471] = 0;
    p[472] = 0;
    p[473] = 0;
    p[474] = 0;
    p[475] = 0;
    p[476] = 0;
    p[477] = 0;
    p[478] = 0;
    p[479] = 0;
    p[480] = 0;
    p[481] = 0;
    p[482] = 0;
    p[483] = 0;
    p[484] = 0;
    p[485] = 0;
    p[486] = 0;
    p[487] = 0;
    p[488] = 0;
    p[489] = 0;
    p[490] = 0;
    p[491] = 0;
    p[492] = 0;
    p[493] = 0;
    p[494] = 0;
    p[495] = 0;
    p[496] = 0;
    p[497] = 0;
    p[498] = 0;
    p[499] = 0;
    p[500] = 0;
    p[501] = 0;
    p[502] = 0;
    p[503] = 0;
    p[504] = 0;
    p[505] = 0;
    p[506] = 0;
    p[507] = 0;
    p[508] = 0;
    p[509] = 0;
    p[510] = 0;
    p[511] = 0;
    p[512] = 0;
    p[513] = 0;
    p[514] = 0;
    p[515] = 0;
    p[516] = 0;
    p[517] = 0;
    p[518] = 0;
    p[519] = 0;
    p[520] = 0;
    p[521] = 0;
    p[522] = 0;
    p[523] = 0;
    p[524] = 0;
    p[525] = 0;
    p[526] = 0;
    p[527] = 0;
    p[528] = 0;
    p[529] = 0;
    p[530] = 0;
    p[531] = 0;
    p[532] = 0;
    p[533] = 0;
    p[534] = 0;
    p[535] = 0;
    p[536] = 0;
    p[537] = 0;
    p[538] = 0;
    p[539] = 0;
    p[540] = 0;
    p[541] = 0;
    p[542] = 0;
    p[543] = 0;
    p[544] = 0;
    p[545] = 0;
    p[546] = 0;
    p[547] = 0;
    p[548] = 0;
    p[549] = 0;
    p[550] = 0;
    p[551] = 0;
    p[552] = 0;
    p[553] = 0;
    p[554] = 0;
    p[555] = 0;
    p[556] = 0;
    p[557] = 0;
    p[558] = 0;
    p[559] = 0;
    p[560] = 0;
    p[561] = 0;
    p[562] = 0;
    p[563] = 0;
    p[564] = 0;
    p[565] = 0;
    p[566] = 0;
    p[567] = 0;
    p[568] = 0;
    p[569] = 0;
    p[570] = 0;
    p[571] = 0;
    p[572] = 0;
    p[573] = 0;
    p[574] = 0;
    p[575] = 0;
    p[576] = 0;
    p[577] = 0;
    p[578] = 0;
    p[579] = 0;
    p[580] = 0;
    p[581] = 0;
    p[582] = 0;
    p[583] = 0;
    p[584] = 0;
    p[585] = 0;
    p[586] = 0;
    p[587] = 0;
    p[588] = 0;
    p[589] = 0;
    p[590] = 0;
    p[591] = 0;
    p[592] = 0;
    p[593] = 0;
    p[594] = 0;
    p[595] = 0;
    p[596] = 0;
    p[597] = 0;
    p[598] = 0;
    p[599] = 0;
    p[600] = 0;
    p[601] = 0;
    p[602] = 0;
    p[603] = 0;
    p[604] = 0;
    p[605] = 0;
    p[606] = 0;
    p[607] = 0;
    p[608] = 0;
    p[609] = 0;
    p[610] = 0;
    p[611] = 0;
    p[612] = 0;
    p[613] = 0;
    p[614] = 0;
    p[615] = 0;
    p[616] = 0;
    p[617] = 0;
    p[618] = 0;
    p[619] = 0;
    p[620] = 0;
    p[621] = 0;
    p[622] = 0;
    p[623] = 0;
    p[624] = 0;
    p[625] = 0;
    p[626] = 0;
    p[627] = 0;
    p[628] = 0;
    p[629] = 0;
    p[630] = 0;
    p[631] = 0;
    p[632] = 0;
    p[633] = 0;
    p[634] = 0;
    p[635] = 0;
    p[636] = 0;
    p[637] = 0;
    p[638] = 0;
    p[639] = 0;
    p[640] = 0;
    p[641] = 0;
    p[642] = 0;
    p[643] = 0;
    p[644] = 0;
    p[645] = 0;
    p[646] = 0;
    p[647] = 0;
    p[648] = 0;
    p[649] = 0;
    p[650] = 0;
    p[651] = 0;
    p[652] = 0;
    p[653] = 0;
    p[654] = 0;
    p[655] = 0;
    p[656] = 0;
    p[657] = 0;
    p[658] = 0;
    p[659] = 0;
    p[660] = 0;
    p[661] = 0;
    p[662] = 0;
    p[663] = 0;
    p[664] = 0;
    p[665] = 0;
    p[666] = 0;
    p[667] = 0;
    p[668] = 0;
    p[669] = 0;
    p[670] = 0;
    p[671] = 0;
    p[672] = 0;
    p[673] = 0;
    p[674] = 0;
    p[675] = 0;
    p[676] = 0;
    p[677] = 0;
    p[678] = 0;
    p[679] = 0;
    p[680] = 0;
    p[681] = 0;
    p[682] = 0;
    p[683] = 0;
    p[684] = 0;
    p[685] = 0;
    p[686] = 0;
    p[687] = 0;
    p[688] = 0;
    p[689] = 0;
    p[690] = 0;
    p[691] = 0;
    p[692] = 0;
    p[693] = 0;
    p[694] = 0;
    p[695] = 0;
    p[696] = 0;
    p[697] = 0;
    p[698] = 0;
    p[699] = 0;
    p[700] = 0;
    p[701] = 0;
    p[702] = 0;
    p[703] = 0;
    p[704] = 0;
    p[705] = 0;
    p[706] = 0;
    p[707] = 0;
    p[708] = 0;
    p[709] = 0;
    p[710] = 0;
    p[711] = 0;
    p[712] = 0;
    p[713] = 0;
    p[714] = 0;
    p[715] = 0;
    p[716] = 0;
    p[717] = 0;
    p[718] = 0;
    p[719] = 0;
    p[720] = 0;
    p[721] = 0;
    p[722] = 0;
    p[723] = 0;
    p[724] = 0;
    p[725] = 0;
    p[726] = 0;
    p[727] = 0;
    p[728] = 0;
    p[729] = 0;
    p[730] = 0;
    p[731] = 0;
    p[732] = 0;
    p[733] = 0;
    p[734] = 0;
    p[735] = 0;
    p[736] = 0;
    p[737] = 0;
    p[738] = 0;
    p[739] = 0;
    p[740] = 0;
    p[741] = 0;
    p[742] = 0;
    p[743] = 0;
    p[744] = 0;
    p[745] = 0;
    p[746] = 0;
    p[747] = 0;
    p[748] = 0;
    p[749] = 0;
    p[750] = 0;
    p[751] = 0;
    p[752] = 0;
    p[753] = 0;
    p[754] = 0;
    p[755] = 0;
    p[756] = 0;
    p[757] = 0;
    p[758] = 0;
    p[759] = 0;
    p[760] = 0;
    p[761] = 0;
    p[762] = 0;
    p[763] = 0;
    p[764] = 0;
    p[765] = 0;
    p[766] = 0;
    p[767] = 0;
    p[768] = 0;
    p[769] = 0;
    p[770] = 0;
    p[771] = 0;
    p[772] = 0;
    p[773] = 0;
    p[774] = 0;
    p[775] = 0;
    p[776] = 0;
    p[777] = 0;
    p[778] = 0;
    p[779] = 0;
    p[780] = 0;
    p[781] = 0;
    p[782] = 0;
    p[783] = 0;
    p[784] = 0;
    p[785] = 0;
    p[786] = 0;
    p[787] = 0;
    p[788] = 0;
    p[789] = 0;
    p[790] = 0;
    p[791] = 0;
    p[792] = 0;
    p[793] = 0;
    p[794] = 0;
    p[795] = 0;
    p[796] = 0;
    p[797] = 0;
    p[798] = 0;
    p[799] = 0;
    p[800] = 0;
    p[801] = 0;
    p[802] = 0;
    p[803] = 0;
    p[804] = 0;
    p[805] = 0;
    p[806] = 0;
    p[807] = 0;
    p[808] = 0;
    p[809] = 0;
    p[810] = 0;
    p[811] = 0;
    p[812] = 0;
    p[813] = 0;
    p[814] = 0;
    p[815] = 0;
    p[816] = 0;
    p[817] = 0;
    p[818] = 0;
    p[819] = 0;
    p[820] = 0;
    p[821] = 0;
    p[822] = 0;
    p[823] = 0;
    p[824] = 0;
    p[825] = 0;
    p[826] = 0;
    p[827] = 0;
    p[828] = 0;
    p[829] = 0;
    p[830] = 0;
    p[831] = 0;
    p[832] = 0;
    p[833] = 0;
    p[834] = 0;
    p[835] = 0;
    p[836] = 0;
    p[837] = 0;
    p[838] = 0;
    p[839] = 0;
    p[840] = 0;
    p[841] = 0;
    p[842] = 0;
    p[843] = 0;
    p[844] = 0;
    p[845] = 0;
    p[846] = 0;
    p[847] = 0;

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
