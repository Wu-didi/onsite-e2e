class PathPoint:
    def __init__(self, pp_list):
        # pp_list: from CalcRefLine, [rx, ry, rs, rtheta, rkappa, rdkappa] x y 路程 角度 角度变化量/路程变化量 (角度变化量/路程变化量)/路程变化量
        self.rx = pp_list[0]
        self.ry = pp_list[1]
        self.rs = pp_list[2]
        self.rtheta = pp_list[3]
        self.rkappa = pp_list[4]
        self.rdkappa = pp_list[5]