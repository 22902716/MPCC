import numpy as np

class dataSave:
    def __init__(self, TESTMODE, map_name,max_iter):
        self.rowSize = 5000
        self.stateCounter = 0
        self.lapInfoCounter = 0
        self.TESTMODE = TESTMODE
        self.map_name = map_name
        self.max_iter = max_iter
        self.txt_x0 = np.zeros((self.rowSize,8))
        self.txt_lapInfo = np.zeros((max_iter,8))

    def saveStates(self, time, x0, expected_speed, tracking_error, noise, completion):
        self.txt_x0[self.stateCounter,0] = time
        self.txt_x0[self.stateCounter,1] = completion
        self.txt_x0[self.stateCounter,2:5] = [x0[0],x0[1],x0[3]]
        self.txt_x0[self.stateCounter,5] = expected_speed
        self.txt_x0[self.stateCounter,6] = tracking_error
        self.txt_x0[self.stateCounter,7] = noise
        self.stateCounter += 1

    def savefile(self, iter):
        for i in range(self.rowSize):
            if (self.txt_x0[i,4] == 0):
                self.txt_x0 = np.delete(self.txt_x0, slice(i,self.rowSize),axis=0)
                break
        np.savetxt(f"Imgs/{self.map_name}/{self.TESTMODE}/{str(iter)}.csv", self.txt_x0, delimiter = ',', header="laptime, completion, ego_x_pos, ego_y_pos, actual speed, expected speed, tracking error", fmt="%-10f")
        self.txt_x0 = np.zeros((self.rowSize,8))
        self.stateCounter = 0
    
    def lapInfo(self,lap_count, lap_success, laptime, completion, var1, var2, aveTrackErr, Computation_time):
        self.txt_lapInfo[self.lapInfoCounter, 0] = lap_count
        self.txt_lapInfo[self.lapInfoCounter, 1] = lap_success
        self.txt_lapInfo[self.lapInfoCounter, 2] = laptime
        self.txt_lapInfo[self.lapInfoCounter, 3] = completion
        self.txt_lapInfo[self.lapInfoCounter, 4] = var1
        self.txt_lapInfo[self.lapInfoCounter, 5] = var2
        self.txt_lapInfo[self.lapInfoCounter, 6] = aveTrackErr
        self.txt_lapInfo[self.lapInfoCounter, 7] = Computation_time
        self.lapInfoCounter += 1

    def saveLapInfo(self):
        if self.TESTMODE == "Benchmark":
            var1 = "NA"
            var2 = "NA"
        if self.TESTMODE == "perception_noise" or self.TESTMODE == "Outputnoise_speed" or self.TESTMODE == "Outputnoise_steering":
            var1 = "noise_scale"
            var2 = "max_noise(m)"
        if self.TESTMODE == "control_delay_speed" or self.TESTMODE == "control_Delay_steering" or self.TESTMODE == "perception_delay":
            var1 = "delay time"
            var2 = "NA"
        if self.TESTMODE == "v_gain":
            var1 = "v_gain"
            var2 = "lookahead dist"
        if self.TESTMODE == "lfd":
            var1 = "lfd_constant"
            var2 = "lookahead dist"
          
        np.savetxt(f"csv/{self.map_name}/{self.map_name}_{self.TESTMODE}.csv", self.txt_lapInfo,delimiter=',',header = f"lap_count, lap_success, laptime, completion, {var1}, {var2}, NA, Computation_time", fmt="%-10f")