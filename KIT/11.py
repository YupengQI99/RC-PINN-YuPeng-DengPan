from collections import defaultdict, deque


def find_min_bus_routes(start, breakfast, end, routes):
    # 构建站台到路线的邻接表
    station_to_routes = defaultdict(set)
    for i, route in enumerate(routes):
        for station in route:
            station_to_routes[station].add(i)

    # 使用广度优先搜索 (BFS)
    def bfs(start_station, target_station):
        queue = deque([(start_station, 0)])  # (当前站台, 已乘坐的公交车数量)
        visited_stations = set([start_station])
        visited_routes = set()

        while queue:
            station, bus_count = queue.popleft()

            # 如果到达目标站台
            if station == target_station:
                return bus_count

            # 遍历经过当前站台的所有路线
            for route_id in station_to_routes[station]:
                if route_id in visited_routes:
                    continue

                visited_routes.add(route_id)

                # 在该路线中，所有站台都可以访问
                for next_station in routes[route_id]:
                    if next_station not in visited_stations:
                        visited_stations.add(next_station)
                        queue.append((next_station, bus_count + 1))

        return -1  # 无法到达目标站台

    # 首先找从上车站到早餐站台的最小公交数
    to_breakfast = bfs(start, breakfast)
    if to_breakfast == -1:
        return -1

    # 然后从早餐站台到公司站台
    to_end = bfs(breakfast, end)
    if to_end == -1:
        return -1

    return to_breakfast + to_end


# 输入
if __name__ == "__main__":
    # 读取第一行
    start, breakfast, end = map(int, input().split())

    # 读取公交路线数量
    num_routes = int(input())

    # 读取公交路线信息
    routes = []
    for _ in range(num_routes):
        route_info = list(map(int, input().split()))
        routes.append(route_info[1:])  # 跳过路线的站台数量

    # 调用函数并输出结果
    result = find_min_bus_routes(start, breakfast, end, routes)
    print(result)