import numpy as np
import scipy
import scipy.signal
import scipy.fftpack
import pyworld.pyworld as pw

def wave2world(x,fs):
    # if gpu_acceralation:
    #     cp=importlib.import_module('cupy')
    # else:
    #     cp=numpy
    f0 = dio(x,fs)
    # sp = get_spectram_envelove(x,f0,fs)
    # ap = get_aperiodicity(x,f0,fs)
    # return f0,sp,ap
def world2wave(f0,sp,ap,fs):
    wave = None
    return wave

log_2=0.69314
epsilon=1e-12
def dio(x:np.array,
        fs:int,
        period:int=5,
        f0_floor:float=71.0,
        f0_ceil:float=800.0,
        ch_in_oct:int=2,
        speed:int= 11,
        allowed_range:float=0.1) -> np.array:
    # initialization of constants
    x_length=len(x)
    n_bands = 1+int(np.log2(f0_ceil/f0_floor)*ch_in_oct)
    boundary_f0 = f0_floor * np.power(2.0,np.arange(1,n_bands+1)/ch_in_oct)
    # normalization
    decimation_ratio=  int(np.clip(speed,1,12))
    y_length = x_length//decimation_ratio+2
    actual_fs = fs/decimation_ratio
    sample = y_length + (4 * int( 1 + actual_fs / boundary_f0[0] / 2))
    fft_size = 2**int(np.log2(sample)+1)
    # decimation(downsampling) to fft
    ys=np.zeros([y_length])
    if decimation_ratio!=1:
        kNFact = 9
        tmp1 = np.zeros([x_length + kNFact * 2])
        tmp2 = np.zeros([x_length + kNFact * 2])

        for i in range(kNFact):
            tmp1[i] = 2 * x[0] - x[kNFact - i]
        for i in range(kNFact, kNFact + x_length):
            tmp1[i] = x[i - kNFact]
        for i in range(kNFact + x_length, 2 * kNFact + x_length):
            tmp1[i] = 2 * x[x_length - 1] - x[x_length - 2 - (i - (kNFact + x_length))]
        FilterForDecimate(tmp1, 2 * kNFact + x_length, decimation_ratio, tmp2)
        for i in range( 2 * kNFact + x_length):
            tmp1[i] = tmp2[2 * kNFact + x_length - i - 1]
        FilterForDecimate(tmp1, 2 * kNFact + x_length,  decimation_ratio, tmp2)
        for i in range(2 * kNFact + x_length):
            tmp1[i] = tmp2[2 * kNFact + x_length - i - 1]
        nout = (x_length - 1) // decimation_ratio + 1
        nbeg =  decimation_ratio -  decimation_ratio * nout + x_length

        count = 0
        for  i in range(nbeg,(x_length + kNFact)-1, decimation_ratio) :
            ys[count] = tmp1[i + kNFact - 1]
            count += 1
    else:
        ys=x
    
    mean_y=np.mean(ys)
    ys-=mean_y
    ys[y_length:]=0.0
    y=scipy.fftpack.fft(ys,n=fft_size)
    y=np.array([y.real,-y.imag])
    y[0,-1]=y[0,1]
    y[1,0]=0.0
    y[1,-1]=0.0

    cut_off= int(np.round(actual_fs/50)* 2 + 1)
    low_cut_filter=ys.copy()
    r=np.arange(1,cut_off+1)*2*np.pi/(cut_off+1)
    low_cut_filter[:cut_off]=0.5-0.5*np.cos(r)
    sum_of_aptitude = np.sum(low_cut_filter[:cut_off])
    low_cut_filter[:cut_off] = -low_cut_filter[:cut_off]/sum_of_aptitude
    low_cut_filter[-((cut_off-1)//2)+1:]=low_cut_filter[:((cut_off-1)//2-1)][::-1]
    low_cut_filter[:cut_off]=low_cut_filter[(cut_off-1)//2:(cut_off-1)//2+cut_off]
    low_cut_filter[0]+=1.0
    
    y_filter=scipy.fftpack.fft(low_cut_filter,n=fft_size)
    y_filter=np.array([y_filter.real,-y_filter.imag])
    y_filter[0,-1]=y_filter[0,1]
    y_filter[1,0]=0.0
    y_filter[1,-1]=0.0

    y_spectrum=np.zeros([fft_size,2])
    y_spectrum[:,0]=y[0]*y_filter[0]-y[1]*y_filter[1]
    y_spectrum[:,1]=y[0]*y_filter[1]+y[1]*y_filter[0]
    f0_length=int(1000*len(x)/fs/period)+1
    f0_candidates=np.zeros([n_bands,f0_length])
    f0_scores=np.zeros([n_bands,f0_length])
    tmp_positions=np.arange(f0_length) * period / 1000
    # f0 estimation
    # getting filtered signal
    
    for j in range(n_bands):

        nega_i_l = np.zeros([y_length])
        nega_i   = np.zeros([y_length])
        posi_i_l = np.zeros([y_length])
        posi_i   = np.zeros([y_length])
        peak_i_l = np.zeros([y_length])
        peak_i   = np.zeros([y_length])
        dips_i_l = np.zeros([y_length])
        dips_i   = np.zeros([y_length])
        
        half_range_length=np.round(fs/boundary_f0[j]/2)
        low_pass_filter=np.zeros([fft_size])
        tmp=np.arange(half_range_length*4)/(half_range_length*4-1)*2*np.pi
        low_pass_filter[:int(half_range_length*4)]=0.355768-0.487396*np.cos(tmp)+0.144232*np.cos(2*tmp)-0.012604*np.cos(3*tmp)
        low_pass_filter_spec_tmp=scipy.fftpack.fft(low_pass_filter,n=fft_size)

        low_pass_filter_spec_tmp=[low_pass_filter_spec_tmp.real,-low_pass_filter_spec_tmp.imag]
        low_pass_filter_spec_tmp[0][-1]=low_pass_filter_spec_tmp[0][1]
        low_pass_filter_spec_tmp[1][0]=0.0
        low_pass_filter_spec_tmp[1][-1]=0.0
        low_pass_filter_spec=np.zeros([fft_size,2])

        half_fft_size=fft_size//2
        ## convolution
        low_pass_filter_spec[:half_fft_size,0]=y_spectrum[:half_fft_size,0]*low_pass_filter_spec_tmp[0][:half_fft_size] \
                                                -y_spectrum[:half_fft_size,1]*low_pass_filter_spec_tmp[1][:half_fft_size]
        low_pass_filter_spec[:half_fft_size,1]=y_spectrum[:half_fft_size,0]*low_pass_filter_spec_tmp[1][:half_fft_size] \
                                                +y_spectrum[:half_fft_size,1]*low_pass_filter_spec_tmp[0][:half_fft_size]

        low_pass_filter_spec[half_fft_size:]=low_pass_filter_spec[:half_fft_size-1:-1]
        low_pass_filter_spec[half_fft_size:,1]*=-1
        p_in=low_pass_filter_spec.reshape(-1)

        filtered_signal=scipy.fftpack.irfft(p_in,n=fft_size).real*2
        # Compensation of the delay
        index_bias=int(half_range_length*2)
        filtered_signal[:y_length]=filtered_signal[index_bias:index_bias+y_length]
        
        # F0 CandidateCounter
        zc_number_of_negatives=zero_crossing_engine(filtered_signal,fft_size,actual_fs,nega_i_l,nega_i)
        zc_number_of_positives=zero_crossing_engine(-1*filtered_signal,fft_size,actual_fs,posi_i_l,posi_i)
        delta = filtered_signal[:-1] - filtered_signal[1:]
        zc_number_of_peaks=zero_crossing_engine(delta,fft_size-1,actual_fs,peak_i_l,peak_i)
        zc_number_of_dips=zero_crossing_engine(-delta,fft_size-1,actual_fs,dips_i_l,dips_i)
        interpolated_f0_set=np.zeros([4,f0_length])
        r=(zc_number_of_negatives-2>0)*(zc_number_of_positives-2>0)*(zc_number_of_peaks-2>0)*(zc_number_of_dips-2>0) == 0
        if r:
            f0_scores[j,:]=np.full([y_length],100000)
            f0_candidates[j,:]=np.zeros([y_length])
            continue
        interpolated_f0_set[0]=interp1(nega_i_l,nega_i,zc_number_of_negatives,tmp_positions,f0_length)
        interpolated_f0_set[1]=interp1(posi_i_l,posi_i,zc_number_of_positives,tmp_positions,f0_length)
        interpolated_f0_set[2]=interp1(peak_i_l,peak_i,zc_number_of_peaks,tmp_positions,f0_length)
        interpolated_f0_set[3]=interp1(dips_i_l,dips_i,zc_number_of_dips,tmp_positions,f0_length)
        
        f0_candidates[j]=(interpolated_f0_set[0]+interpolated_f0_set[1]+interpolated_f0_set[2]+interpolated_f0_set[3])/4
        i0=interpolated_f0_set[0,j]-f0_candidates[j]
        i1 = interpolated_f0_set[0, j] - f0_candidates[j]
        i2 = interpolated_f0_set[0, j] - f0_candidates[j]
        i3 = interpolated_f0_set[0, j] - f0_candidates[j]
        f0_scores[j]=np.sqrt((i0*i1+i1*i1+i2*i2+i3*i3)/3)
        for i in range(f0_length):
            if f0_candidates[j,i] > boundary_f0[j] or f0_candidates[j,i] < boundary_f0[j] / 2 or f0_candidates[j,i] > f0_ceil or f0_candidates[j,i] <f0_floor:
                f0_scores[j,i]=100000
                f0_candidates[j,i]=0 

    f0_scores=f0_scores/(f0_candidates+epsilon)
    # getting best f0 counter
    c=0
    best_f0_counter=np.zeros([f0_scores.shape[1]])
    for n in np.argmin(f0_scores,axis=0):
        best_f0_counter[c] = f0_candidates[n,c]
        c+=1

    # fix f0 counter with 4 steps
    voice_range_minimum=int(0.5+1000/period/f0_floor)
    # step1
    f0_tmp1=np.zeros([f0_length])
    f0_base=np.zeros([f0_length])
    f0_base[voice_range_minimum:-voice_range_minimum]=best_f0_counter[voice_range_minimum:-voice_range_minimum]
    for i in range(voice_range_minimum,f0_length):
        f0_tmp1[i]= f0_base[i] if abs((f0_base[i]-f0_base[i-1])/(epsilon+f0_base[i]))<allowed_range else 0
    # step2
    f0_tmp2=f0_tmp1.copy()
    center=(voice_range_minimum-1)//2
    for i in range(center,f0_length-center):
        for j in range(-center,center):
            if f0_tmp1[i+j]==0:
                f0_tmp2[i]=0
                break
    positive_index=np.zeros([f0_length],dtype=np.int16)
    negative_index=np.zeros([f0_length],dtype=np.int16)
    pcount=0
    ncount=0
    f0_tmp1=f0_tmp2.copy()
    for i in range(1,f0_length):
        if f0_tmp1[i]==0 and f0_tmp1[i-1]!=0:
            negative_index[ncount]=i-1
            ncount+=1
        elif f0_tmp1[i]!=0 and f0_tmp1[i-1]==0:
            positive_index[pcount]=i
            pcount+=1
    # step3
    for i in range(ncount):
        limit = f0_length-1 if i==ncount-1 else negative_index[i+1]
        for j in range(negative_index[i],limit):
            f0_tmp2[j+1]=select_best_f0(f0_tmp1[j],f0_tmp1[j-1],f0_candidates,n_bands,j+1,allowed_range)
            if f0_tmp2[j+1]==0:
                break
    f0_tmp1=f0_tmp2.copy()
    f0=np.zeros([f0_length])
    # step4
    for i in range(pcount-1,0,-1):
        limit = 1 if i==0 else positive_index[i-1]
        for j in range(positive_index[i],limit,-1):
            f0[j-1]=select_best_f0(f0_tmp1[j],f0_tmp1[j+1],f0_candidates,n_bands,j-1,allowed_range)
            if f0[j-1]==0:
                break
    
    return f0

def FilterForDecimate(x,x_length,r,y):
    
    a_0=[0.0,0.041156,0.9503937898,1.4499664, 1.76109396, 1.9715352,2.122523,2.23574623,2.3236003,2.393647,2.450743,2.498139]
    a_1=[0.0,-0.42599,-0.6742914674,-0.9894349,-1.25549,-1.4686795,-1.6395,-1.77808999,-1.8921545,-1.98739,-2.067949,-2.13689]
    a_2=[0.0,0.041037,0.15412211,0.2457825,0.323718,0.38939,0.444697,0.49152555,0.53148928,0.565887,0.595747,0.621875]
    b_0=[0.0,0.167974,0.071221945,0.0367107,0.021334,0.013469,0.00903,0.006352,0.004633,0.00348186,0.002682,0.002109]
    b_1=[0.0,0.503923,0.21366583,0.1101322,0.0640045,0.04040,0.027110,0.0190568,0.0138993,0.010445,0.008046,0.006329]

    a=[a_0[r],a_1[r],a_2[r]]
    b=[b_0[r],b_1[r]]
    w = [0.0, 0.0, 0.0]
    for i in range(x_length):
        wt = x[i] + a[0] * w[0] + a[1] * w[1] + a[2] * w[2]
        y[i] = b[0] * wt + b[1] * w[0] + b[1] * w[1] + b[0] * w[2]
        w[2] = w[1]
        w[1] = w[0]
        w[0] = wt
    return 0

def select_best_f0(cur_f0,past_f0,f0_candidates,number_of_candidates,target_index,allowed_range):
    ref_f0=(cur_f0*3.0-past_f0)/2.0
    min_err=np.abs(ref_f0-f0_candidates[0][target_index])
    best_f0=f0_candidates[0][target_index]
    for  i in range(1,number_of_candidates):
        cur_err=np.abs(ref_f0-f0_candidates[i][target_index])
        if cur_err<min_err:
            min_err=cur_err
            best_f0=f0_candidates[i][target_index]
    if np.abs(1.0-best_f0/ref_f0)>allowed_range:
        return 0.0
    return best_f0
def interp1(x,y,xl,xi,xil):
    h=x[1:]-x[:-1]
    k=np.zeros([xil],dtype=np.int16)
    
    # histc
    i=0
    stage=0
    count=1
    while i < xil:
        if stage==0:
            k[i]=1
            if xi[i] >= x[0]:
                stage=1
        elif stage==1:
            if xi[i] < x[count]:
                k[i]=count
            else:
                k[i] = count
                i-=1
                count+=1           
            if count==xl:
                break
        i+=1
    count-=1
    i+=1
    while i < xil:
        k[i]=count
        i+=1
    #end histc
    yi=np.zeros([xil])
    for i in range(xil):
        s=(xi[i]-x[k[i]-1]) / h[k[i]-1]
        yi[i]=y[k[i]-1]+s*(y[k[i]]-y[k[i]-1])
    return yi
def zero_crossing_engine(signal,y_length,actual_fs,interval_locations,intervals):
    negative_going_points=np.zeros([y_length])
    for i in range(y_length-1):
        negative_going_points[i]= i+1 if 0.0 < signal[i] and signal[i+1] <= 0 else 0
    negative_going_points[-1]=0
    edges=np.zeros([y_length],dtype=np.int16)
    counts=0
    for i in range(y_length):
        if negative_going_points[i]>0:
            counts+=1
            edges[counts]=negative_going_points[i]
    if counts<=2:
        return 0
    fine_edges=np.zeros([counts])
    for i in range(counts):
        fine_edges[i]=edges[i]-signal[edges[i]-1]/(signal[edges[i]]-signal[edges[i]-1])
    for i in range(counts-1):
        intervals[i]=actual_fs/(fine_edges[i+1]-fine_edges[i])
        interval_locations[i]=(fine_edges[i]+fine_edges[i+1]) /2.0 / actual_fs
    return counts-1

import wave
import matplotlib.pyplot as plt
if __name__=="__main__":
    CHUNK=1024
    dms=[]
    wf = wave.open("dataset/test/test.wav", 'rb')
    dds = wf.readframes(CHUNK)
    while dds != b'':
        dms.append(dds)
        dds = wf.readframes(CHUNK)
    dms = b''.join(dms)
    data = np.frombuffer(dms, 'int16')
    data_real=(data/32767).reshape(-1).astype(np.float)
    data_realA=dmn=data_real.copy()
    f0,t=pw.dio(data_realA,16000)
    f0_alpha=dio(data_realA,16000)
    plt.plot(f0)
    plt.plot(f0_alpha)
    plt.show()