# JHU-LLM-Project
Final Project for the LLM class

Current status:

Nov 19: A working mechanism that mutates and judges prompt, given a task.

Improvements:

1. Add CI into the score. This helps avoid the situation where a nascent gene takes over a good old gene because of CI. Best to use score - CI for judgement.
2. Load other types of model, not just the current 7 billion parameter model (Done).

Test indices: 1018, 319, 62, 140, 378, 440, 32, 1061, 92, 396, 64, 1054, 79, 716, 1284, 623, 1238, 399, 1191, 381, 123, 45, 497, 812, 824, 599, 911, 13, 426, 860, 519, 453, 1119, 406, 931, 358, 560, 807, 1232, 424, 941, 996, 258, 1039, 680, 20, 711, 1124, 651, 114, 1041, 1229, 882, 289, 868, 1162, 99, 983, 511, 888, 518, 48, 738, 628, 59, 966, 777, 292, 1196, 821, 660, 644, 545, 321, 190, 329, 400, 1267, 770, 784, 361, 988, 850, 788, 1315, 1163, 210, 142, 1206, 1156, 444, 549, 833, 347, 49, 1231, 422, 1265, 159, 581