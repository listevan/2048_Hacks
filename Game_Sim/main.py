import game

def runTerminal():
    g = game.Game()
    g.display()
    while True:
        print("--------------------------------------")
        print("0. UP")
        print("1. DOWN")
        print("2. RIGHT")
        print("3. LEFT")
        print("INPUT DIRECTION: ")
        dir = input()
        print("-------------------------------------")
        if dir.lstrip().rstrip() not in {'0', '1', '2', '3'}:
            print('go fuck yourself')
            continue

        print("SELECTED DIR:", dir)
        
        g.move(int(dir))
        g.display()

if __name__ == '__main__':
    runTerminal()