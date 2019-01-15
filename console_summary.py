import struct
import time
import json
import matplotlib.pyplot as plot

class ConsoleSummary():
    def __init__(self, file=None):
        self.results = {}
        self.iteration = []
        self.file = file
        pass

    def add_summary(self, summary, iteration):
        try:
            result = {
                "time": time.time(),
                "iteration": iteration,
            }

            self.iteration.append(iteration)
            # summaryはバイナリ配列
            # \x20 ? \x20 [1]name_length [可変]variable_name \x15 [4]value という構造をしているため、それをパースする
            parse = []
            pos = 0
            while pos < len(summary):
                _t = summary[pos:pos + 4]
                name_length = summary[pos + 3]
                pos += 4
                name = summary[pos:pos+name_length].decode()
                pos += 1 + name_length
                value = struct.unpack('<f', summary[pos:pos+4])[0]
                parse.append((_t, name, value))
                pos += 4

            for _, name, value in parse:
                result[name] = value
                if name in self.results:
                    self.results[name].append(value)
                else:
                    self.results[name] = [value]

            if self.file:
                with LoggingJSON(self.file) as f:
                    f.append(result)
            
            print("--- Summary ---")
            padding = max([len(a) for a in self.results.keys()]) + 1

            for k in self.results:
                print("%s: " % k.rjust(padding, ' '), end='')
                prev = False
                for v in self.results[k][-5:]:
                    if prev:
                        print("-> %f (%+f) " % (v , v - prev), end='')
                    else:
                        print("%f " % v, end='')
                    prev = v
                print()
                
            print("----------------")
        except:
            print("Summary fail")
            import traceback
            traceback.print_exc()
            print(summary)

    def plot(self):
        data = LoggingJSON(self.file).all()
        plot.clf()
        count = len(data.keys())
        i = 0
        for k, v in data.items():
            i += 1
            plot.subplot(count, 1, i)
            plot.plot(v)
            plot.title(k)

        plot.show()
        

    

class LoggingJSON():
    def __init__(self, file):
        self.file = file

    def __enter__(self):
        self.fp = open(self.file, 'a+')
        return self
    
    def __exit__(self, exception_type, exception_value, traceback):
        self.fp.close()

    def append(self, dict):
        self.fp.write(json.dumps(dict) + '\n', separators=(',', ':'))

    def all(self):
        with open(self.file, 'r') as f:
            dataarray = map(lambda line: json.loads(line), f.readlines())
            data = {}
            for d in dataarray:
                for k, v in d.items():
                    if k in data:
                        data[k].append(v)
                    else:
                        data[k] = [v]
        
        return data


