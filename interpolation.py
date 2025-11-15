import numpy as np
import matplotlib.pyplot as plt
import json
import glob
from tqdm.auto import tqdm

def interpolate(A: np.ndarray,B: np.ndarray,f: float) -> np.ndarray:
    return((1-f)*A + f*B)
def read_json(json_file) -> dict:
    with open(json_file,"r") as f:
        data = json.load(f)
        f.close()
    return(data)

def aspect_to_coords(aspect_str)-> np.ndarray:
    positives = ["Beyond","Heat","Light","Life","Prodigiouss","Energy","Extrospection","Flexibility","Hope","Movement"]
    negatives = ["Within","Cold","Dark","Death","Diminutive","Potential","Introspection","Rigidity","Hopelessness","Stagnency"]
    coords = np.zeros(len(positives))
    if aspect_str in positives:
        coords[positives.index(aspect_str)] = 1
        return(coords)
    elif aspect_str in negatives:
        coords[negatives.index(aspect_str)] = -1
        return(coords)
    elif "Num" in aspect_str or "Dist" in aspect_str or isinstance(aspect_str,list):
        return(coords)
    else:
        raise KeyError(f"{aspect_str} not in {positives}\n or {negatives}")

def attribute_to_coords(aspect_list) -> np.ndarray:
    coords = np.zeros(10)
    for aspect_str in aspect_list:
        coords += aspect_to_coords(aspect_str)
    return(coords)

def get_all_attribute_coords(json_file):
    test_json = read_json(json_file)
    att_keys = test_json.keys()
    att_coords = []
    for ak in att_keys:
        att_coords.append(attribute_to_coords(test_json[ak]))
    return(att_coords,att_keys)

def test_interpolation(coord0,coord1,inter_density = 1000,threshold = 0.25):
    passed_points = []
    err = []
    F = []
    for f in np.linspace(0,1,inter_density):
        new_point = interpolate(coord0,coord1,f)
        new_point_int = np.round(new_point,0)
        err_ = np.sum(np.abs(new_point - new_point_int))/10
        if err_ < threshold:
            
            if len(passed_points) > 0:
                if np.min(np.sqrt(np.sum(np.square(new_point_int - np.array(passed_points)),axis = 1))) != 0:
                    passed_points.append(new_point_int)
                    err.append(err_*10)
                    F.append(f)
            else:
                passed_points.append(new_point_int)
                err.append(err_)
                F.append(f)
    return(passed_points[1:-1],err[1:-1],F[1:-1])


def get_nearest_f(A,B,interp_point):
    # Mostly based on https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    # section "Vector Formulation"
    
    #get normal vector, n
    n = B - A
    n = n/np.linalg.norm(n)
    
    #get difference between initial point and interpolation point
    # this is a vector from P to A
    diff = (A - interp_point)

    #get distance between the nearest point on line AB to P, and A
    # np.dot(diff,n) is the projection of thje line PA to the line AB
    # multiply by the normal vector "n" to turn this into a vector so that 
    # (diff.n)*n is the lenth on the line AB
    length = np.dot(diff,n)*n

    #Get the distance from P to the projected line on AB
    # To understand this we can imagine these as vector points
    # where the projected line + (diff - length) would give P
    D = np.linalg.norm(diff - length)
    
    #get point C, this is just to get the interpolation point that's actually 
    # on the line.
    C = A - length #negative because A length is a vector from C -> A

    #get the non-zero indices so that when dividing we don't get nans
    # non_zero_CA should = non_zero_BA, might be worth adding a check 
    # for that in future
    non_zero_CA = np.where(C-A != 0)

    #get f of minimum D
    f = np.mean(np.abs((C-A)[non_zero_CA]/(B-A)[non_zero_CA])) #should be the same for all though, this is not like...totally correct
                                        # but should be correct enough for this
    return(D,f)
    
if __name__ == "__main__":
    test_json = read_json(r"aspects\dtype_aspects.json")
    dtype1 = attribute_to_coords(test_json["Lightning"])
    dtype2 = attribute_to_coords(test_json["None"])
    print("Lightning: ",dtype1)
    print("None: ",dtype2)
    passed_points,err,F = test_interpolation(dtype1,dtype2)
    print("---------------")
    for i,pp in enumerate(passed_points):
        D,f = get_nearest_f(dtype1,dtype2,pp)
        print(f"interpolated coord: {pp} \t @ f= {f:.2f}\t ΔE= {D:.4f}")
print(f"test_jeon)