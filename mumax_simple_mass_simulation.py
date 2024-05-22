import subprocess
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import math
import csv

path = str(pathlib.Path(__file__).parent.resolve()) + "\\mumax3.exe"#"C:\Users\gavin\Desktop\micromag_sim\mumax3.exe"
print(path)

"""
Built in assumptions: 
1. The hysterisis loop is computed along the x direction
2. I do substring testing on the header information, this could be mangled
3. We iterate from the positive to the negative. There is no other way
4. The coercivity must be contained in the range or we will except
5. I'm adjusting the sheet parameters, so be sure of units
"""


def parseCoercivityFromFile(name, displayPlot = False):
    with open(name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        header = next(csv_reader)

        m = 0
        b = 0
        j = 0
        for i in header:
            if "mx" in i:
                m = j
            if "B_extx" in i:
                b = j
            j += 1
        
        #Now, loop through the rows:

        data = list(map(lambda a: list(map(float, a)) , list(csv_reader)))

        if displayPlot:
            npData = np.array(data)

            plt.plot(npData[:, b], npData[:, m])
            plt.xlabel("B")
            plt.ylabel("M")
            plt.title("Hysteresis Loop")
            plt.show()
            plt.plot()

        for row in range(len(data) - 1):
            if data[row][m] * data[row+1][m] <= 0.0:
                #Now measure the linear interp to get the best val possible

                gap = math.fabs(data[row][m] - data[row+1][m])
                rowp0 = math.fabs(data[row][m] / gap)
                rowp1 = math.fabs(data[row+1][m] / gap)
                coercivity = rowp0 * data[row+1][b] + rowp1 * data[row][b]#switch quantities
                return math.fabs(coercivity)
    
    print("something's wrong")
    return -1#Negative indicates something is wrong

def createFileString(msat, aex, ku1):
    #Note that I have to adjust the parameters from the sheet manually
    return """
//setup some variables := for declaration, = for setting the value
N:=32
c:=1e-9 //in meters


//Initialize the mesh and the scale of the simulation
setgridsize(N, N, N)
setcellsize(c, c, c)


//Add the variables which will be recorded for each time step
TableAdd(B_ext) //external field

Msat = """+str(msat)+ """ // A/m  //uniformly distributed Msat
Aex = """+str(aex)+"""e-12 // J/m //unifom
Ku1 = """+str(ku1)+"""e6 // J/m3 //uniform
Temp = 300
alpha = 0.03//Hysteresis loop is independent of this
anisU = vector(1, 0, 0)

//simulation


//This is to allow the simulation to run faster
B_ext = vector(6, 0, 0)
relax()//This is for if -change in B_ext- is large

B_ext = vector(4, 0, 0)
relax()//This is for if -change in B_ext- is large

TableSave()


//hysteresis loop for coercivity

for bbb:=4.0; bbb > 0.0; bbb+=-0.1{
    B_ext = vector(bbb, 0, 0)

    minimize() //this is for if -change in B_ext- is small

    TableSave()
}

relax()

for bbb:=0.0; bbb > -1.0; bbb+=-0.01{
    if m.Comp(0).average() >= 0.0 { //apparently the x-component
        B_ext = vector(bbb, 0, 0)

        minimize()

        TableSave()
    }
}

relax()//for numerical stability

for bbb:=-1.0; bbb > -2.0; bbb+=-0.01{
    if m.Comp(0).average() >= 0.0 { //apparently the x-component
        B_ext = vector(bbb, 0, 0)

        minimize()

        TableSave()
    }
}

relax()//for numerical stability

for bbb:=-2.0; bbb > -3.0; bbb+=-0.01{
    if m.Comp(0).average() >= 0.0 { //apparently the x-component
        B_ext = vector(bbb, 0, 0)

        minimize()

        TableSave()
    }
}
"""

def runMagSim(msat, aex, ku1, ii):
    with open("template" + str(ii) + ".txt", "w") as file:
        prog = createFileString(msat, aex, ku1)
        file.write(prog)
    
    #Note that mumax has to be on the PATH
    cmd = [path, "template" + str(ii) + ".txt"]
    result = subprocess.run(cmd)

    print(result.returncode)
    return "template" + str(ii) + ".out\\table.txt"

with open('DFTData.csv', 'r') as csvIn:
    reader = csv.reader(csvIn)
    with open('ml_mumax.csv', 'w', newline='', encoding='utf-8') as csvOut:
        writer = csv.writer(csvOut)

        jj = 0

        for row in reader:
            jj+=1
            ouputFile = runMagSim(row[0], row[1], row[2], jj)
            coercivity = parseCoercivityFromFile(ouputFile)

            #m a k h order
            writer.writerow((row[0], row[1], row[2], coercivity))

#Current approach
#ML:  Aex, Msat, Ku1 -> Hc (exp)

#Past approach - alpha is important, maybe test for alpha
# (1/3 [cube]) * (? [material]) = alpha
#ML: Aex, Msat, Ku1 -> Hc (mumax)
#validate on
#ML: Aex, Msat, Ku1 -> Hc (exp)

#New Approach - data augmentation
#MUMAX: Aex, Msat, Ku1 -> Hc (mumax)
#ML: Aex, Msat, Ku1, Hc (mumax) -> Hc (exp)

#New Approach - data finetuning
#first training
#ML: Aex, Msat, Ku1 -> Hc (mumax)
#then second training
#ML: Aex, Msat, Ku1 -> Hc (exp)