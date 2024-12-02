import display_stuff
import serial
import umyo_parser
from serial.tools import list_ports

ser = serial.Serial(
    port="COM7",
    baudrate=921600,
    parity=serial.PARITY_NONE,
    stopbits=1,
    bytesize=8,
    timeout=0,
)

print("conn: " + ser.portstr)
last_data_upd = 0
display_stuff.plot_init()
parse_unproc_cnt = 0
while 1:
    cnt = ser.in_waiting
    if cnt > 0:
        cnt_corr = parse_unproc_cnt / 200
        data = ser.read(cnt)
        parse_unproc_cnt = umyo_parser.umyo_parse_preprocessor(data)
        dat_id = display_stuff.plot_prepare(umyo_parser.umyo_get_list())
        d_diff = 0
        if not (dat_id is None):
            d_diff = dat_id - last_data_upd
        if d_diff > 2 + cnt_corr:
            display_stuff.plot_cycle_tester()
            last_data_upd = dat_id
