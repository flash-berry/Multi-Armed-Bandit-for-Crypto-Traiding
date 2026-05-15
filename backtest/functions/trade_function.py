def count_usdt_final(final_asset_quantity, close_price, trade_cost):
    """Рассчитывает сумму выхода из сделки с учётом издержки в одну сторону"""
    asset_held = final_asset_quantity
    usdt_from_sale = asset_held * close_price
    fee_in_usdt = usdt_from_sale * trade_cost
    usdt_final = usdt_from_sale - fee_in_usdt

    return usdt_final

def count_entry_action(position_value, close_price, trade_cost):
    """Рассчитывает сумму входа в сделку с учётом издержки в одну сторону"""
    asset_bought = position_value / close_price
    fee_in_asset = asset_bought * trade_cost
    final_asset_quantity = asset_bought - fee_in_asset

    return final_asset_quantity