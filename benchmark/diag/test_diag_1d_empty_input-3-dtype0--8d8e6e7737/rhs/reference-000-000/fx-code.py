


def forward(self, arg0_1):
    diag_embed = torch.ops.aten.diag_embed.default(arg0_1, 3);  arg0_1 = None
    return diag_embed
    
