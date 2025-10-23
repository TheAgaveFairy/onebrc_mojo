from random import seed
from sys.info import num_logical_cores
from sys import stderr
from time import perf_counter_ns
import os
import benchmark
from pathlib import Path

alias FILENAME = "../data/measurements.txt"

struct Stats(Copyable & Movable & ImplicitlyCopyable):
    # range of possible temps is [-99.9, 99.9]. to avoid floats, we'll do a *= 10
    alias min_value = -999
    alias max_value =  999

    var lowest: Int
    var highest: Int
    var total: Int
    var count: Int

    fn __init__(out self):
        self.lowest = Self.max_value
        self.highest = Self.min_value
        self.total = 0
        self.count = 0

    fn update(mut self, temp: Int):
        self.lowest = min(self.lowest, temp)
        self.highest = max(self.highest, temp)
        self.total += temp
        self.count += 1

    fn getMean(self) -> String:
        var whole = abs(self.total) // (self.count * 10) # dont forget to undo the *= 10
        var remainder = abs(self.total) % (self.count * 10)

        var sign = -1
        if self.total >= 0:
            sign = 1
        #if remainder < 0:
        #    remainder = -remainder

        var dec = (remainder * 10 + self.count * 10 - 1) // (self.count * 10) # ceiling div
        if sign < 0:
            dec = (remainder * 10) // (self.count * 10)

        #if dec >= 10:
        #    whole += sign
        #    dec = 0
        #whole *= sign
        var sign_str = "-" if sign == -1 else ""
        # TODO: ensure this enforces roundTowardsPositive IEEE-754 behavior
        return sign_str + String(whole) + '.' + String(dec)

    fn __str__(self) -> String:
        # return "=" + String(self.lowest) + "/" + String(self.getMean()) + "/" + String(self.highest)
        return "[" + String(self.lowest) + ", " + String(self.highest) + "] : " + String(self.total) + "/" + String(self.count) + " total/count.\n"

    @staticmethod
    fn getTemp(text: String) -> Int:
        """
        Our own parsing, unsafe + no floating point! Nice!
        Returns int equal to the temp *= 10.
        Ex. Input: '-36.9' gives Output: -369.
        """
        var negative: Int = 1
        var i: Int = 0
        var ans: Int = 0

        if text[0] == '-':
            negative = -1
            i += 1

        while i < len(text):
            if text[i] == '.':
                i += 1
                continue
            ans += ord(text[i]) - ord('0')
            i += 1
            ans *= 10

        ans //= 10
        ans *= negative
        return ans

    fn __copyinit__(out self, other: Self): # trivial
        self.lowest = other.lowest
        self.highest = other.highest
        self.total = other.total
        self.count = other.count

    fn __moveinit__(out self, deinit existing: Self): # trivial
        self.lowest = existing.lowest
        self.highest = existing.highest
        self.total = existing.total
        self.count = existing.count

@fieldwise_init
struct TestResult(Copyable & Movable & ImplicitlyCopyable):
    alias __copyinit__is_trivial = True
    alias __moveinit__is_trivial = True
    var buffer_size: Int
    var semicolons: Int
    var elapsed_ns: Int

    fn __str__(self) -> String:
        return "Buffer Size (B): " + String(self.buffer_size)
            + " Elapsed ms: " + String(self.elapsed_ns // 1_000_000)
            + " ';' found: " + String(self.semicolons)

fn readFileTest() -> List[TestResult]:
    alias KB = 1024
    var buffer_sizes_kb = InlineArray[Int, 6](64, 256, 1024, 4096, 8192, 16384)
    
    var results = List[TestResult](capacity = len(buffer_sizes_kb))
    
    try:
        var stat = Path(FILENAME).stat()
        var file_len = stat.st_size
        print(file_len, "B", FILENAME)

        for buf_sz_idx in range(len(buffer_sizes_kb)):
            var start_time = perf_counter_ns()
            var buf_sz = buffer_sizes_kb[buf_sz_idx] * KB
            var idx = 0
            var count = 0 # counting semicolons for workload
            with open(FILENAME, "r") as f:
                while idx < file_len:
                    var text = f.read_bytes(buf_sz)
                    #if len(text) != buf_sz:
                        #print("leftovers :: ", len(text), buf_sz)
                    for i in range(len(text)): # some ~arbitrary work to accomplish
                        if text[i] == ord(';'):
                            count += 1
                    idx += buf_sz

                var end_time = perf_counter_ns()
                var elapsed_ns = (end_time - start_time) // 1_000_000
                print(elapsed_ns, "ns")
                var result = TestResult(buf_sz, count, elapsed_ns)
                results[buf_sz_idx] = result
                count = 0
                idx = 0

                print(result.__str__())
    except e:
        print(e)
    return results^

def main():
    var start_time = perf_counter_ns()
    
    #var report = benchmark.run[readFileTest]()
    var results = readFileTest()
    for result in results:
        print("---", result.__str__())
    """
    var map = Dict[String, Stats]()

    var num_lines = 0
    with open("../data/measurements.txt", "r") as f:
        var text = f.read()
        var lines = text.split('\n') # UTF-8 reminder

        #lines = lines[:10_000] # LIMIT

        num_lines = len(lines)
        for line in lines:
            var pair = line.split(';')
            # print(pair.__str__())
            var station = String(pair[0])
            var temp = Stats.getTemp(String(pair[1]))

            if station in map:
                map[station].update(temp)
            else:
                map[station] = Stats()
                map[station].update(temp)

    for item in map.items():
        var station = item.key
        var stats = item.value
        #if stats.count == 3:
            #print(station, stats.__str__(), stats.getMean())
    """

    var end_time = perf_counter_ns()
    var elapsed_ms = (end_time - start_time) / 1_000_000
